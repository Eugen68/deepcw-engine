import fs from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";
import * as ort from "onnxruntime-node";

const MIN_AUDIO_SECONDS = 5.0;
const MAX_AUDIO_SECONDS = 20.0;

function parseArgs() {
  const args = {
    model: "../../model.onnx",
    metadata: "../../model.onnx.json",
    wav: null,
  };

  for (let i = 2; i < process.argv.length; i += 1) {
    const key = process.argv[i];
    const value = process.argv[i + 1];
    if (key === "--model" && value) args.model = value;
    if (key === "--metadata" && value) args.metadata = value;
    if (key === "--wav" && value) args.wav = value;
    if (key.startsWith("--")) i += 1;
  }

  if (!args.wav) {
    throw new Error("Missing required argument: --wav <path-to-5-to-20-second-pcm-wav>");
  }

  return args;
}

function readString(buffer, offset, length) {
  return buffer.toString("ascii", offset, offset + length);
}

function readWavMono(filePath) {
  const buffer = fs.readFileSync(filePath);
  if (readString(buffer, 0, 4) !== "RIFF" || readString(buffer, 8, 4) !== "WAVE") {
    throw new Error("Expected a RIFF/WAVE file.");
  }

  let offset = 12;
  let format = null;
  let dataOffset = -1;
  let dataSize = 0;

  while (offset + 8 <= buffer.length) {
    const chunkId = readString(buffer, offset, 4);
    const chunkSize = buffer.readUInt32LE(offset + 4);
    const chunkData = offset + 8;

    if (chunkId === "fmt ") {
      format = {
        audioFormat: buffer.readUInt16LE(chunkData),
        channels: buffer.readUInt16LE(chunkData + 2),
        sampleRate: buffer.readUInt32LE(chunkData + 4),
        bitsPerSample: buffer.readUInt16LE(chunkData + 14),
      };
    } else if (chunkId === "data") {
      dataOffset = chunkData;
      dataSize = chunkSize;
    }

    offset = chunkData + chunkSize + (chunkSize % 2);
  }

  if (!format || dataOffset < 0) {
    throw new Error("The WAV file is missing a fmt or data chunk.");
  }
  if (format.audioFormat !== 1) {
    throw new Error(`Only PCM WAV is supported by this compact example. Got format ${format.audioFormat}.`);
  }
  if (![8, 16, 32].includes(format.bitsPerSample)) {
    throw new Error(`Unsupported PCM bit depth: ${format.bitsPerSample}`);
  }

  const bytesPerSample = format.bitsPerSample / 8;
  const frameCount = Math.floor(dataSize / (bytesPerSample * format.channels));
  const audio = new Float32Array(frameCount);

  for (let frame = 0; frame < frameCount; frame += 1) {
    let sum = 0;
    for (let channel = 0; channel < format.channels; channel += 1) {
      const sampleOffset = dataOffset + (frame * format.channels + channel) * bytesPerSample;
      if (format.bitsPerSample === 8) sum += (buffer.readUInt8(sampleOffset) - 128) / 128;
      if (format.bitsPerSample === 16) sum += buffer.readInt16LE(sampleOffset) / 32768;
      if (format.bitsPerSample === 32) sum += buffer.readInt32LE(sampleOffset) / 2147483648;
    }
    audio[frame] = sum / format.channels;
  }

  return { sampleRate: format.sampleRate, audio };
}

function resampleLinear(audio, sourceRate, targetRate) {
  if (sourceRate === targetRate) return audio;
  const targetLength = Math.round((audio.length * targetRate) / sourceRate);
  const output = new Float32Array(targetLength);

  for (let i = 0; i < targetLength; i += 1) {
    const sourcePosition = (i * sourceRate) / targetRate;
    const left = Math.floor(sourcePosition);
    const right = Math.min(left + 1, audio.length - 1);
    const fraction = sourcePosition - left;
    output[i] = audio[left] * (1 - fraction) + audio[right] * fraction;
  }

  return output;
}

function frequencyBinRange(sampleRate, fftLength, minHz, maxHz) {
  const binHz = sampleRate / fftLength;
  return {
    startBin: Math.ceil(minHz / binHz),
    stopBin: Math.floor(maxHz / binHz) + 1,
  };
}

function selectedDftMagnitudes(frame, startBin, stopBin) {
  const length = frame.length;
  const output = new Float32Array(stopBin - startBin);

  for (let bin = startBin; bin < stopBin; bin += 1) {
    let real = 0;
    let imaginary = 0;
    for (let n = 0; n < length; n += 1) {
      const angle = (-2 * Math.PI * bin * n) / length;
      real += frame[n] * Math.cos(angle);
      imaginary += frame[n] * Math.sin(angle);
    }
    output[bin - startBin] = Math.hypot(real, imaginary);
  }

  return output;
}

export function audioToSpectrogram(audio, metadata) {
  const fftLength = metadata.fft_length;
  const hopLength = metadata.hop_length;
  const expectedBins = metadata.spectrogram_frequency_bins;
  if (audio.length < fftLength) throw new Error(`Audio is too short for fft_length=${fftLength}.`);

  const { startBin, stopBin } = frequencyBinRange(
    metadata.sample_rate,
    fftLength,
    metadata.spectrogram_min_freq_hz,
    metadata.spectrogram_max_freq_hz,
  );
  if (stopBin - startBin !== expectedBins) {
    throw new Error(`Metadata expects ${expectedBins} bins, but computed ${stopBin - startBin}.`);
  }

  const pad = Math.floor(fftLength / 2);
  const padded = new Float32Array(audio.length + pad * 2);
  for (let i = 0; i < pad; i += 1) {
    padded[i] = audio[pad - i];
    padded[pad + audio.length + i] = audio[audio.length - 2 - i];
  }
  padded.set(audio, pad);

  const frames = 1 + Math.floor((padded.length - fftLength) / hopLength);
  const tensorData = new Float32Array(frames * expectedBins);
  const window = new Float32Array(fftLength);
  for (let i = 0; i < fftLength; i += 1) {
    window[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / fftLength);
  }

  const frame = new Float32Array(fftLength);
  for (let frameIndex = 0; frameIndex < frames; frameIndex += 1) {
    const start = frameIndex * hopLength;
    for (let i = 0; i < fftLength; i += 1) {
      frame[i] = padded[start + i] * window[i];
    }
    const magnitudes = selectedDftMagnitudes(frame, startBin, stopBin);
    for (let bin = 0; bin < expectedBins; bin += 1) {
      tensorData[frameIndex * expectedBins + bin] = Math.log1p(magnitudes[bin]);
    }
  }

  return new ort.Tensor("float32", tensorData, [1, 1, frames, expectedBins]);
}

function greedyCtcDecode(logProbs, chars, blankIndex) {
  const [batch, frames, classes] = logProbs.dims;
  if (batch !== 1) throw new Error(`Expected batch size 1, got ${batch}.`);

  let previous = null;
  let decoded = "";
  for (let frame = 0; frame < frames; frame += 1) {
    let bestIndex = 0;
    let bestValue = -Infinity;
    for (let klass = 0; klass < classes; klass += 1) {
      const value = logProbs.data[frame * classes + klass];
      if (value > bestValue) {
        bestValue = value;
        bestIndex = klass;
      }
    }

    if (bestIndex === blankIndex) {
      previous = null;
    } else {
      if (bestIndex !== previous) decoded += chars[bestIndex];
      previous = bestIndex;
    }
  }

  return decoded;
}

async function main() {
  const args = parseArgs();
  const modelPath = path.resolve(process.cwd(), args.model);
  const metadataPath = path.resolve(process.cwd(), args.metadata);
  const wavPath = path.resolve(process.cwd(), args.wav);
  const metadata = JSON.parse(fs.readFileSync(metadataPath, "utf8"));
  const { sampleRate, audio } = readWavMono(wavPath);
  const resampled = resampleLinear(audio, sampleRate, metadata.sample_rate);
  const durationSeconds = resampled.length / metadata.sample_rate;

  if (durationSeconds < MIN_AUDIO_SECONDS || durationSeconds > MAX_AUDIO_SECONDS) {
    throw new Error(
      `Audio duration must be between ${MIN_AUDIO_SECONDS} and ${MAX_AUDIO_SECONDS} seconds after resampling. ` +
        `Got ${durationSeconds.toFixed(2)} seconds.`,
    );
  }

  const spectrogram = audioToSpectrogram(resampled, metadata);
  const session = await ort.InferenceSession.create(modelPath);
  const outputs = await session.run({ [metadata.onnx_input_name]: spectrogram });
  const decoded = greedyCtcDecode(outputs[metadata.onnx_output_name], metadata.chars, metadata.blank_index);
  console.log(decoded);
}

if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) {
  main().catch((error) => {
    console.error(error);
    process.exitCode = 1;
  });
}

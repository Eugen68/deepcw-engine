// Continuous DeepCW reception with overlapping windows.
//
// The model is trained on clips of 5-20 seconds, so a long stream is cut into
// overlapping windows. Each window is decoded to CTC tokens stamped with an
// absolute sample time and a confidence. Consecutive windows are merged by
// aligning the tokens in their reliable overlap, so each character is emitted
// exactly once and the text is never cut in the middle of a character.
//
//   ffmpeg -i input.wav -ac 1 -ar 3200 -f s16le - | node cw_stream.mjs

import fs from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";
import * as ort from "onnxruntime-node";

import { audioToSpectrogram } from "./decode_morse.mjs";

export const DEFAULT_CONFIG = Object.freeze({
  windowSeconds: 12.0,
  hopSeconds: 8.0,
  // Slow CW characters last seconds; a window that starts or ends inside one can misread it.
  edgeGuardSeconds: 1.0,
  matchToleranceSeconds: 0.1,
});

// Greedy CTC over one window. Returns { sample (frame, possibly .5), index, score } per run.
export function ctcGreedyTokens(data, frames, classes, blankIndex) {
  const tokens = [];
  let runIndex = -1;
  let runStart = 0;
  let runPeak = -Infinity;
  const close = (stop) => {
    if (runIndex >= 0 && runIndex !== blankIndex) {
      tokens.push({ sample: (runStart + stop - 1) * 0.5, index: runIndex, score: Math.exp(runPeak) });
    }
  };
  for (let frame = 0; frame < frames; frame += 1) {
    let bestIndex = 0;
    let bestValue = -Infinity;
    for (let klass = 0; klass < classes; klass += 1) {
      const value = data[frame * classes + klass];
      if (value > bestValue) {
        bestValue = value;
        bestIndex = klass;
      }
    }
    if (bestIndex !== runIndex) {
      close(frame);
      runIndex = bestIndex;
      runStart = frame;
      runPeak = bestValue;
    } else if (bestValue > runPeak) {
      runPeak = bestValue;
    }
  }
  close(frames);
  return tokens;
}

// Time-constrained alignment of two short token sequences; see cw_stream.py.
function align(a, b, tolerance) {
  const n = a.length;
  const m = b.length;
  const gap = 1.0;
  const cost = Array.from({ length: n + 1 }, () => new Float64Array(m + 1).fill(Infinity));
  const move = Array.from({ length: n + 1 }, () => new Int8Array(m + 1)); // 0 pair, 1 a-only, 2 b-only
  for (let i = 0; i <= n; i += 1) {
    cost[i][0] = i * gap;
    move[i][0] = 1;
  }
  for (let j = 0; j <= m; j += 1) {
    cost[0][j] = j * gap;
    move[0][j] = 2;
  }
  for (let i = 1; i <= n; i += 1) {
    for (let j = 1; j <= m; j += 1) {
      let best = cost[i - 1][j] + gap;
      let step = 1;
      if (cost[i][j - 1] + gap < best) {
        best = cost[i][j - 1] + gap;
        step = 2;
      }
      const dt = Math.abs(a[i - 1].sample - b[j - 1].sample);
      if (dt <= tolerance) {
        const pair = cost[i - 1][j - 1] + (0.5 * dt) / tolerance + (a[i - 1].index === b[j - 1].index ? 0 : 0.9);
        if (pair <= best) {
          best = pair;
          step = 0;
        }
      }
      cost[i][j] = best;
      move[i][j] = step;
    }
  }
  const pairs = [];
  let i = n;
  let j = m;
  while (i || j) {
    const step = move[i][j];
    if (step === 0) {
      pairs.push([a[i - 1], b[j - 1]]);
      i -= 1;
      j -= 1;
    } else if (step === 1) {
      pairs.push([a[i - 1], null]);
      i -= 1;
    } else {
      pairs.push([null, b[j - 1]]);
      j -= 1;
    }
  }
  return pairs.reverse();
}

// Merge token streams of overlapping windows into one committed stream; see cw_stream.py.
export class OverlapMerger {
  constructor(edgeGuard, tolerance) {
    this.edgeGuard = edgeGuard;
    this.tolerance = tolerance;
    this.pending = [];
    this.pendingReliableEnd = -Infinity;
    this.cutoff = -Infinity;
    this.windows = 0;
  }

  push(windowStart, windowLength, windowTokens) {
    const reliableStart = windowStart + (this.windows ? this.edgeGuard : 0);
    const tokens = windowTokens.filter((token) => token.sample >= reliableStart);
    this.windows += 1;
    if (this.windows === 1) {
      this.pending = tokens;
      this.pendingReliableEnd = windowStart + windowLength - this.edgeGuard;
      return [];
    }

    let zoneStart = Math.max(reliableStart, this.cutoff);
    let zoneEnd = this.pendingReliableEnd;
    if (zoneEnd < zoneStart) {
      // No reliable overlap: trust each window up to the middle of the gap.
      zoneStart = zoneEnd = 0.5 * (zoneStart + zoneEnd);
    }
    const middle = 0.5 * (zoneStart + zoneEnd);
    const inZone = (token) => zoneStart <= token.sample && token.sample <= zoneEnd;

    const committed = this.pending.filter((token) => token.sample < zoneStart);
    const merged = [];
    for (const [old, now] of align(this.pending.filter(inZone), tokens.filter(inZone), this.tolerance)) {
      if (old && now) merged.push(old.score >= now.score ? old : now);
      else if (old && old.sample < middle) merged.push(old);
      else if (now && now.sample >= middle) merged.push(now);
    }
    // Unpaired tokens between two pairs come back in arbitrary a/b order.
    merged.sort((x, y) => x.sample - y.sample);
    committed.push(...merged);

    this.cutoff = zoneEnd;
    this.pending = tokens.filter((token) => token.sample > zoneEnd);
    this.pendingReliableEnd = windowStart + windowLength - this.edgeGuard;
    return committed;
  }

  finish() {
    const committed = this.pending;
    this.pending = [];
    return committed;
  }
}

// Push audio chunks of any size; receive merged tokens as they become final.
// Audio must already be mono at the model sample rate.
export class StreamingDecoder {
  constructor(session, metadata, config = DEFAULT_CONFIG) {
    const rate = metadata.sample_rate;
    this.session = session;
    this.metadata = metadata;
    this.outputHop = metadata.hop_length; // one output frame per spectrogram frame
    this.window = Math.round(config.windowSeconds * rate);
    // Keep window starts on the output-frame grid so spikes of different windows are comparable.
    this.hop = Math.max(this.outputHop, Math.round((config.hopSeconds * rate) / this.outputHop) * this.outputHop);
    const edgeGuard = Math.round(config.edgeGuardSeconds * rate);
    const tolerance = config.matchToleranceSeconds * rate;
    if (this.hop > this.window || this.window - this.hop - 2 * edgeGuard < 2 * tolerance) {
      throw new Error("Window overlap minus both edge guards is too short to align the windows.");
    }
    this.merger = new OverlapMerger(edgeGuard, tolerance);

    this.buffer = new Float32Array(this.window + this.hop);
    this.fill = 0; // valid samples in buffer
    this.bufferStart = 0; // absolute sample index of buffer[0]
    this.nextWindow = 0; // absolute start of the next window
    this.lastWindowEnd = 0;
  }

  get totalSamples() {
    return this.bufferStart + this.fill;
  }

  async push(audio) {
    const committed = [];
    let position = 0;
    while (position < audio.length) {
      if (this.fill === this.buffer.length) this.compact();
      const take = Math.min(this.buffer.length - this.fill, audio.length - position);
      this.buffer.set(audio.subarray(position, position + take), this.fill);
      this.fill += take;
      position += take;
      while (this.nextWindow + this.window <= this.totalSamples) {
        const offset = this.nextWindow - this.bufferStart;
        committed.push(...(await this.decode(this.nextWindow, this.buffer.slice(offset, offset + this.window))));
        this.lastWindowEnd = this.nextWindow + this.window;
        this.nextWindow += this.hop;
      }
    }
    return committed;
  }

  // Decode the remaining tail and return every outstanding token.
  async flush() {
    const total = this.totalSamples;
    const committed = [];
    if (total > this.lastWindowEnd && total >= this.metadata.fft_length) {
      // The tail window ends at the stream end; a short stream is decoded whole.
      const start = Math.max(0, total - this.window);
      committed.push(
        ...(await this.decode(start, this.buffer.slice(start - this.bufferStart, total - this.bufferStart))),
      );
    }
    return committed.concat(this.merger.finish());
  }

  compact() {
    // Drop samples no future window needs, keeping enough for a tail window at flush.
    const keepFrom = Math.min(this.nextWindow, this.totalSamples - this.window);
    const drop = keepFrom - this.bufferStart;
    this.buffer.copyWithin(0, drop, this.fill);
    this.fill -= drop;
    this.bufferStart = keepFrom;
  }

  async decode(start, audio) {
    const spectrogram = audioToSpectrogram(audio, this.metadata);
    const outputs = await this.session.run({ [this.metadata.onnx_input_name]: spectrogram });
    const logProbs = outputs[this.metadata.onnx_output_name];
    const [, frames, classes] = logProbs.dims;
    const tokens = ctcGreedyTokens(logProbs.data, frames, classes, this.metadata.blank_index).map((token) => ({
      ...token,
      sample: start + Math.round(token.sample * this.outputHop),
    }));
    return this.merger.push(start, audio.length, tokens);
  }
}

export function tokensToText(tokens, chars) {
  return tokens.map((token) => chars[token.index]).join("");
}

function parseArgs() {
  const args = { model: "../../model.onnx", metadata: "../../model.onnx.json", ...DEFAULT_CONFIG };
  const numeric = {
    "--window-seconds": "windowSeconds",
    "--hop-seconds": "hopSeconds",
    "--edge-guard-seconds": "edgeGuardSeconds",
  };
  for (let i = 2; i < process.argv.length; i += 1) {
    const key = process.argv[i];
    const value = process.argv[i + 1];
    if (key === "--model" && value) args.model = value;
    if (key === "--metadata" && value) args.metadata = value;
    if (key in numeric && value) args[numeric[key]] = Number(value);
    if (key.startsWith("--")) i += 1;
  }
  return args;
}

async function main() {
  const args = parseArgs();
  const metadata = JSON.parse(fs.readFileSync(path.resolve(process.cwd(), args.metadata), "utf8"));
  const session = await ort.InferenceSession.create(path.resolve(process.cwd(), args.model), {
    executionProviders: ["cpu"],
    graphOptimizationLevel: "all",
  });
  const decoder = new StreamingDecoder(session, metadata, args);
  const emit = (tokens) => {
    if (tokens.length) process.stdout.write(tokensToText(tokens, metadata.chars));
  };

  let remainder = Buffer.alloc(0);
  for await (const chunk of process.stdin) {
    const data = Buffer.concat([remainder, chunk]);
    const usable = data.length - (data.length % 2);
    remainder = data.subarray(usable);
    const audio = new Float32Array(usable / 2);
    for (let i = 0; i < audio.length; i += 1) audio[i] = data.readInt16LE(2 * i) / 32768;
    emit(await decoder.push(audio));
  }
  emit(await decoder.flush());
  process.stdout.write("\n");
}

if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) {
  main().catch((error) => {
    console.error(error);
    process.exitCode = 1;
  });
}

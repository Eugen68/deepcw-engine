# deepcw-engine

This repository contains the [DeepCW](https://github.com/e04/web-deep-cw-decoder) model and a minimal example.

## Preparing WAV Audio

The examples can read and resample ordinary PCM WAV files, but preparing the audio with
`ffmpeg` makes the input explicit and repeatable:

```bash
ffmpeg -i input.wav -ac 1 -ar 3200 -sample_fmt s16 target.wav
```

The sample rate shown above matches the current `model.onnx.json`. If the metadata changes,
use the `sample_rate` value from that file.

## Python Example

Directory: `examples/python`

Install dependencies:

```bash
cd examples/python
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Run the decoder:

```bash
python decode_morse.py \
  --model ../../model.onnx \
  --metadata ../../model.onnx.json \
  --wav test.wav
```

## Node.js Example

Directory: `examples/nodejs`

Install dependencies:

```bash
cd examples/nodejs
npm install
```

Run the decoder:

```bash
npm run decode -- \
  --model ../../model.onnx \
  --metadata ../../model.onnx.json \
  --wav test.wav
```

## Continuous Decoding

`decode_morse` decodes one 5-20 second clip. For longer recordings or live audio, the
`cw_stream` examples read raw mono 16-bit PCM from stdin and print text as it becomes final:

```bash
cd examples/python
ffmpeg -i input.wav -ac 1 -ar 3200 -f s16le - | python cw_stream.py
```

or

```bash
cd examples/nodejs
ffmpeg -i input.wav -ac 1 -ar 3200 -f s16le - | npm run -s stream
```

The audio is decoded in 12-second windows that start every 8 seconds. Each decoded character
carries its time in the stream and the model's confidence. In the 4-second overlap of two
windows, characters are aligned by time and each is kept once, from the more confident window,
while the outer second of every interior window edge is ignored. A character cut by a window
boundary is therefore read from the window that saw it whole, and nothing is duplicated.
Text appears about 12 seconds after it is received. Window, hop and edge guard can be changed
with `--window-seconds`, `--hop-seconds` and `--edge-guard-seconds`.

On synthetic CW at 12-40 WPM and -3 to +10 dB SNR (2,500 Hz bandwidth), 60-second streams were
decoded without errors, while cutting the same audio into back-to-back 12-second clips gave a
character error rate of 3-10%. The overlap merging follows the stream receiver of
[DeepRTTY](https://github.com/e04/DeepRTTY).

Tests: `cd examples/python && python -m unittest test_cw_stream`

## License

AGPL-3.0-only. See [LICENSE](LICENSE).

"""Continuous DeepCW reception with overlapping windows.

The model is trained on clips of 5-20 seconds, so a long stream is cut into
overlapping windows. Each window is decoded to CTC tokens stamped with an
absolute sample time and a confidence. Consecutive windows are merged by
aligning the tokens in their reliable overlap, so each character is emitted
exactly once and the text is never cut in the middle of a character.

Stream raw mono audio at the model sample rate:

    ffmpeg -i input.wav -ac 1 -ar 3200 -f s16le - | python cw_stream.py
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, NamedTuple

import numpy as np
import onnxruntime as ort

from decode_morse import audio_to_spectrogram, load_metadata


class Token(NamedTuple):
    sample: int  # absolute stream position of the CTC spike, in samples
    index: int  # class index into metadata["chars"]
    score: float  # peak posterior probability of the spike


@dataclass(frozen=True)
class StreamConfig:
    window_seconds: float = 12.0
    hop_seconds: float = 8.0
    # Slow CW characters last seconds; a window that starts or ends inside one can misread it.
    edge_guard_seconds: float = 1.0
    match_tolerance_seconds: float = 0.1


def ctc_greedy_tokens(log_probs: np.ndarray, blank_index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Greedy CTC over one [time, class] matrix.

    Returns run-center frame positions (possibly .5), class indices, and peak
    posteriors. The index sequence equals the one of greedy_ctc_decode.
    """
    path = log_probs.argmax(axis=-1)
    if path.size == 0:
        empty = np.empty(0)
        return empty, empty.astype(np.int64), empty
    best = np.take_along_axis(log_probs, path[:, None], axis=-1)[:, 0]
    starts = np.flatnonzero(np.diff(path, prepend=-1))
    stops = np.append(starts[1:], path.size)
    peaks = np.maximum.reduceat(best, starts)  # one segment per run, blank runs included
    indices = path[starts]
    keep = indices != blank_index
    starts, stops, indices, peaks = starts[keep], stops[keep], indices[keep], peaks[keep]
    return (starts + stops - 1) * 0.5, indices.astype(np.int64), np.exp(peaks.astype(np.float64))


def _align(a: list[Token], b: list[Token], tolerance: float) -> list[tuple[Token | None, Token | None]]:
    """Time-constrained alignment of two short token sequences.

    Pairs are allowed only within `tolerance` samples. Equal characters cost
    less than substitutions, and both cost less than leaving two tokens
    unpaired, so a spurious neighbour cannot steal the partner of a real one.
    """
    n, m = len(a), len(b)
    gap = 1.0
    cost = np.full((n + 1, m + 1), np.inf)
    move = np.zeros((n + 1, m + 1), dtype=np.int8)  # 0 pair, 1 a-only, 2 b-only
    cost[:, 0] = np.arange(n + 1) * gap
    cost[0, :] = np.arange(m + 1) * gap
    move[1:, 0] = 1
    move[0, 1:] = 2
    for i in range(1, n + 1):
        ta, ia = a[i - 1].sample, a[i - 1].index
        for j in range(1, m + 1):
            best, step = cost[i - 1, j] + gap, 1
            if cost[i, j - 1] + gap < best:
                best, step = cost[i, j - 1] + gap, 2
            dt = abs(ta - b[j - 1].sample)
            if dt <= tolerance:
                pair = cost[i - 1, j - 1] + 0.5 * dt / tolerance + (0.0 if ia == b[j - 1].index else 0.9)
                if pair <= best:
                    best, step = pair, 0
            cost[i, j], move[i, j] = best, step
    pairs: list[tuple[Token | None, Token | None]] = []
    i, j = n, m
    while i or j:
        step = move[i, j]
        if step == 0:
            pairs.append((a[i - 1], b[j - 1]))
            i, j = i - 1, j - 1
        elif step == 1:
            pairs.append((a[i - 1], None))
            i -= 1
        else:
            pairs.append((None, b[j - 1]))
            j -= 1
    pairs.reverse()
    return pairs


class OverlapMerger:
    """Merge token streams of overlapping windows into one committed stream.

    Windows must be pushed in order of increasing start. Only the reliable part
    of each window, excluding `edge_guard` samples at interior edges, competes.
    Inside the reliable overlap of two consecutive windows the tokens are
    aligned: a paired token is taken from whichever window is more confident,
    and an unpaired token is kept only by the window that owns its half of the
    overlap. Output is final once returned.
    """

    def __init__(self, edge_guard: int, tolerance: float) -> None:
        self.edge_guard = int(edge_guard)
        self.tolerance = float(tolerance)
        self._pending: list[Token] = []
        self._pending_reliable_end: float = -math.inf
        self._cutoff: float = -math.inf
        self._windows = 0

    def push(self, window_start: int, window_length: int, tokens: Iterable[Token]) -> list[Token]:
        reliable_start = window_start + (self.edge_guard if self._windows else 0)
        tokens = [token for token in tokens if token.sample >= reliable_start]
        self._windows += 1
        if self._windows == 1:
            self._pending = tokens
            self._pending_reliable_end = window_start + window_length - self.edge_guard
            return []

        zone_start = max(reliable_start, self._cutoff)
        zone_end = self._pending_reliable_end
        if zone_end < zone_start:
            # No reliable overlap: trust each window up to the middle of the gap.
            zone_start = zone_end = 0.5 * (zone_start + zone_end)
        middle = 0.5 * (zone_start + zone_end)

        committed = [token for token in self._pending if token.sample < zone_start]
        previous = [token for token in self._pending if zone_start <= token.sample <= zone_end]
        current = [token for token in tokens if zone_start <= token.sample <= zone_end]
        merged: list[Token] = []
        for old, new in _align(previous, current, self.tolerance):
            if old is not None and new is not None:
                merged.append(old if old.score >= new.score else new)
            elif old is not None and old.sample < middle:
                merged.append(old)
            elif new is not None and new.sample >= middle:
                merged.append(new)
        # Unpaired tokens between two pairs come back in arbitrary a/b order.
        merged.sort(key=lambda token: token.sample)
        committed += merged

        self._cutoff = zone_end
        self._pending = [token for token in tokens if token.sample > zone_end]
        self._pending_reliable_end = window_start + window_length - self.edge_guard
        return committed

    def finish(self) -> list[Token]:
        committed, self._pending = self._pending, []
        return committed


class StreamingDecoder:
    """Push audio chunks of any size; receive merged tokens as they become final.

    Audio must already be mono at the model sample rate.
    """

    def __init__(
        self,
        model_path: Path,
        metadata: dict,
        config: StreamConfig = StreamConfig(),
        session: ort.InferenceSession | None = None,
    ) -> None:
        self.metadata = metadata
        rate = int(metadata["sample_rate"])
        self.output_hop = int(metadata["hop_length"])  # one output frame per spectrogram frame
        self.window = round(config.window_seconds * rate)
        # Keep window starts on the output-frame grid so spikes of different windows are comparable.
        self.hop = max(self.output_hop, round(config.hop_seconds * rate / self.output_hop) * self.output_hop)
        edge_guard = round(config.edge_guard_seconds * rate)
        tolerance = config.match_tolerance_seconds * rate
        if self.hop > self.window or self.window - self.hop - 2 * edge_guard < 2 * tolerance:
            raise ValueError("Window overlap minus both edge guards is too short to align the windows.")
        self.merger = OverlapMerger(edge_guard, tolerance)
        self.blank_index = int(metadata["blank_index"])
        self.session = session or create_session(model_path)
        self.input_name = metadata["onnx_input_name"]
        self.output_name = metadata["onnx_output_name"]

        self._buffer = np.zeros(self.window + self.hop, dtype=np.float32)
        self._fill = 0  # valid samples in _buffer
        self._buffer_start = 0  # absolute sample index of _buffer[0]
        self._next_window = 0  # absolute start of the next window
        self._last_window_end = 0

    @property
    def total_samples(self) -> int:
        return self._buffer_start + self._fill

    def push(self, audio: np.ndarray) -> list[Token]:
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        committed: list[Token] = []
        position = 0
        while position < audio.size:
            if self._fill == self._buffer.size:
                self._compact()
            take = min(self._buffer.size - self._fill, audio.size - position)
            self._buffer[self._fill : self._fill + take] = audio[position : position + take]
            self._fill += take
            position += take
            committed += self._cut_windows()
        return committed

    def flush(self) -> list[Token]:
        """Decode the remaining tail and return every outstanding token."""
        total = self.total_samples
        committed: list[Token] = []
        if total > self._last_window_end and total >= int(self.metadata["fft_length"]):
            # The tail window ends at the stream end; a short stream is decoded whole.
            start = max(0, total - self.window)
            committed += self._decode(start, self._buffer[start - self._buffer_start : total - self._buffer_start])
        return committed + self.merger.finish()

    def _cut_windows(self) -> list[Token]:
        committed: list[Token] = []
        while self._next_window + self.window <= self.total_samples:
            offset = self._next_window - self._buffer_start
            committed += self._decode(self._next_window, self._buffer[offset : offset + self.window])
            self._last_window_end = self._next_window + self.window
            self._next_window += self.hop
        return committed

    def _compact(self) -> None:
        # Drop samples no future window needs, keeping enough for a tail window at flush.
        # A full buffer always holds a cut window, so this frees at least one sample.
        keep_from = min(self._next_window, self.total_samples - self.window)
        drop = keep_from - self._buffer_start
        self._buffer[: self._fill - drop] = self._buffer[drop : self._fill]
        self._fill -= drop
        self._buffer_start = keep_from

    def _decode(self, start: int, audio: np.ndarray) -> list[Token]:
        spectrogram = audio_to_spectrogram(audio, self.metadata)
        log_probs = self.session.run([self.output_name], {self.input_name: spectrogram})[0][0]
        frames, indices, scores = ctc_greedy_tokens(log_probs, self.blank_index)
        samples = start + np.rint(frames * self.output_hop).astype(np.int64)
        tokens = [Token(int(s), int(i), float(p)) for s, i, p in zip(samples, indices, scores)]
        return self.merger.push(start, audio.size, tokens)


def create_session(model_path: Path) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(model_path), options, providers=["CPUExecutionProvider"])


def decode_long(
    audio: np.ndarray,
    model_path: Path,
    metadata: dict,
    config: StreamConfig = StreamConfig(),
    session: ort.InferenceSession | None = None,
) -> list[Token]:
    """Decode an arbitrarily long recording at the model sample rate into merged tokens."""
    decoder = StreamingDecoder(model_path, metadata, config, session)
    return decoder.push(audio) + decoder.flush()


def tokens_to_text(tokens: Iterable[Token], chars: list[str]) -> str:
    return "".join(chars[token.index] for token in tokens)


def main() -> None:
    parser = argparse.ArgumentParser(description="Decode continuous CW from raw s16le mono PCM on stdin.")
    parser.add_argument("--model", type=Path, default=Path("../../model.onnx"))
    parser.add_argument("--metadata", type=Path, default=Path("../../model.onnx.json"))
    parser.add_argument("--window-seconds", type=float, default=StreamConfig.window_seconds)
    parser.add_argument("--hop-seconds", type=float, default=StreamConfig.hop_seconds)
    parser.add_argument("--edge-guard-seconds", type=float, default=StreamConfig.edge_guard_seconds)
    args = parser.parse_args()
    metadata = load_metadata(args.metadata)
    config = StreamConfig(args.window_seconds, args.hop_seconds, args.edge_guard_seconds)
    decoder = StreamingDecoder(args.model, metadata, config)
    chars = list(metadata["chars"])

    def emit(tokens: list[Token]) -> None:
        if tokens:
            sys.stdout.write(tokens_to_text(tokens, chars))
            sys.stdout.flush()

    chunk_bytes = 2 * int(metadata["sample_rate"]) // 5
    stdin = sys.stdin.buffer
    remainder = b""
    while data := stdin.read(chunk_bytes):
        data = remainder + data
        usable = len(data) - len(data) % 2
        remainder = data[usable:]
        emit(decoder.push(np.frombuffer(data[:usable], dtype="<i2").astype(np.float32) / 32768.0))
    emit(decoder.flush())
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()

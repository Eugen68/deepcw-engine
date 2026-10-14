"""Tests for cw_stream.py. Run from this directory: python -m unittest test_cw_stream"""

from __future__ import annotations

import math
import unittest
from pathlib import Path

import numpy as np

from cw_stream import OverlapMerger, StreamConfig, StreamingDecoder, Token, create_session, decode_long, tokens_to_text
from decode_morse import audio_to_spectrogram, greedy_ctc_decode, load_metadata

ROOT = Path(__file__).resolve().parents[2]
MORSE = {
    "A": ".-", "B": "-...", "C": "-.-.", "D": "-..", "E": ".", "F": "..-.", "G": "--.", "H": "....",
    "I": "..", "J": ".---", "K": "-.-", "L": ".-..", "M": "--", "N": "-.", "O": "---", "P": ".--.",
    "Q": "--.-", "R": ".-.", "S": "...", "T": "-", "U": "..-", "V": "...-", "W": ".--", "X": "-..-",
    "Y": "-.--", "Z": "--..", "0": "-----", "1": ".----", "2": "..---", "3": "...--", "4": "....-",
    "5": ".....", "6": "-....", "7": "--...", "8": "---..", "9": "----.", "/": "-..-.", "?": "..--..",
}
TEXT = "CQ CQ DE YO3XYZ YO3XYZ K TNX FER CALL UR RST 599 5NN NAME IS ED QTH BUCHAREST HW? BK 73 TU"


def synth_cw(text: str, wpm: float, snr_db: float, seed: int, rate: int = 3200, freq: float = 700.0) -> np.ndarray:
    """Keyed sine with 5 ms raised-cosine edges in white noise; SNR in a 2,500 Hz bandwidth."""
    rng = np.random.default_rng(seed)
    dot = 1.2 / wpm
    units: list[tuple[bool, float]] = [(False, 0.5)]
    for word in text.split():
        for char in word:
            for symbol in MORSE[char]:
                units += [(True, dot * (3 if symbol == "-" else 1)), (False, dot)]
            units.append((False, 2 * dot))
        units.append((False, 4 * dot))
    units.append((False, 0.5))
    key = np.concatenate([np.full(round(seconds * rate), float(on), np.float32) for on, seconds in units])
    edge = np.hanning(2 * round(0.005 * rate) + 1)
    key = np.convolve(key, edge / edge.sum(), mode="same")
    signal = key * np.sin(2 * np.pi * freq * np.arange(key.size) / rate)
    noise = rng.standard_normal(key.size) * math.sqrt(0.5 / 10 ** (snr_db / 10) * (rate / 2) / 2500)
    audio = signal + noise
    return (0.8 * audio / np.abs(audio).max()).astype(np.float32)


class MergerTest(unittest.TestCase):
    def test_overlap_token_is_emitted_once_from_the_more_confident_window(self) -> None:
        merger = OverlapMerger(edge_guard=100, tolerance=50)
        self.assertEqual(merger.push(0, 2000, [Token(500, 1, 0.9), Token(1500, 2, 0.6)]), [])
        committed = merger.push(1000, 2000, [Token(1510, 3, 0.8), Token(2500, 4, 0.9)])
        committed += merger.finish()
        self.assertEqual([(t.sample, t.index) for t in committed], [(500, 1), (1510, 3), (2500, 4)])

    def test_unpaired_token_is_kept_only_by_the_window_owning_its_half(self) -> None:
        merger = OverlapMerger(edge_guard=100, tolerance=50)
        merger.push(0, 2000, [Token(1200, 1, 0.9), Token(1700, 2, 0.9)])  # reliable to 1900
        committed = merger.push(1000, 2000, [Token(1300, 3, 0.9), Token(1800, 4, 0.9)]) + merger.finish()
        # Overlap zone [1100, 1900], middle 1500: the first window owns 1200, the second 1800.
        self.assertEqual([t.index for t in committed], [1, 4])


class StreamTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.metadata = load_metadata(ROOT / "model.onnx.json")
        cls.session = create_session(ROOT / "model.onnx")
        cls.chars = list(cls.metadata["chars"])

    def decode(self, audio: np.ndarray) -> str:
        return " ".join(tokens_to_text(decode_long(audio, None, self.metadata, session=self.session), self.chars).split())

    def test_long_stream_decodes_every_character_once(self) -> None:
        audio = synth_cw(TEXT, wpm=18, snr_db=3, seed=1)
        self.assertGreater(audio.size, 3 * round(12.0 * self.metadata["sample_rate"]))
        self.assertEqual(self.decode(audio), TEXT)

    def test_chunk_size_does_not_change_output(self) -> None:
        audio = synth_cw(TEXT, wpm=25, snr_db=0, seed=2)
        expected = decode_long(audio, None, self.metadata, session=self.session)
        decoder = StreamingDecoder(None, self.metadata, StreamConfig(), session=self.session)
        rng = np.random.default_rng(0)
        tokens, position = [], 0
        while position < audio.size:
            size = int(rng.integers(1, 5000))
            tokens += decoder.push(audio[position : position + size])
            position += size
        self.assertEqual(tokens + decoder.flush(), expected)

    def test_stream_shorter_than_a_window_matches_single_clip_decoder(self) -> None:
        audio = synth_cw("CQ DE YO3XYZ K", wpm=20, snr_db=5, seed=3)
        log_probs = self.session.run(None, {"spectrogram": audio_to_spectrogram(audio, self.metadata)})[0]
        single = greedy_ctc_decode(log_probs, self.chars, int(self.metadata["blank_index"]))
        self.assertEqual(self.decode(audio), " ".join(single.split()))
        self.assertEqual(self.decode(audio), "CQ DE YO3XYZ K")


if __name__ == "__main__":
    unittest.main()

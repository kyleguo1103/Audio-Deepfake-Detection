"""Small, data-free tests for the dashboard's core score path."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from detector import Engine, Probe, sigmoid  # noqa: E402


class FakeModel:
    def __init__(self, offset: float = 0.0):
        self.offset = offset

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        return x.sum(axis=1) + self.offset


class FakePilotCommon:
    @staticmethod
    def resample(signal: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
        if source_rate != target_rate:
            raise AssertionError("This fixture expects 16 kHz input")
        return signal

    @staticmethod
    def preprocess(signal: np.ndarray, source_rate: int, _cfg: dict):
        return signal, {
            "orig_sr": source_rate,
            "lead_silence_s": 0.0,
            "trail_silence_s": 0.0,
            "trim_fallback": False,
        }

    @staticmethod
    def normalise(signal: np.ndarray) -> np.ndarray:
        return signal


class FakeEncoder:
    description = {"model_id": "fake-encoder", "device": "cpu"}

    def __call__(self, _signal: np.ndarray) -> np.ndarray:
        return np.arange(75, dtype=np.float32).reshape(25, 3)


class DetectorCoreTests(unittest.TestCase):
    def test_sigmoid_uses_zero_as_the_decision_midpoint(self):
        self.assertAlmostEqual(sigmoid(0.0), 0.5)
        self.assertGreater(sigmoid(2.0), 0.5)
        self.assertLess(sigmoid(-2.0), 0.5)

    def test_probe_selects_the_requested_hidden_layer(self):
        probe = Probe("none", 2, 1.0, FakeModel(offset=-1.0), 0.0, 0.1, 10)
        embedding = np.arange(15, dtype=np.float32).reshape(5, 3)
        self.assertAlmostEqual(probe.logit(embedding), float(embedding[2].sum() - 1.0))

    def test_analysis_applies_threshold_and_scores_all_probe_groups(self):
        engine = Engine.__new__(Engine)
        engine.pc = FakePilotCommon()
        engine.cfg = {}
        engine.encoder = FakeEncoder()
        engine.cond_summary = {
            "none": {"target_eer_mean": 0.36, "target_balanced_accuracy_mean": 0.50}
        }
        selected = Probe("none", 2, 1.0, FakeModel(offset=-20.0), 0.5, 0.01, 10)
        layer_probe = Probe("none", 4, 1.0, FakeModel(offset=-40.0), 1.0, 0.01, 10)
        engine.probes = {"none": selected}
        engine.layer_probes = {("none", 4): layer_probe}

        result = engine.analyse(np.ones(1600, dtype=np.float32), 16000, "fixture.wav")

        expected_logit = float(np.arange(75, dtype=np.float32).reshape(25, 3)[2].sum() - 20.0)
        self.assertAlmostEqual(result.scores["none"].logit, expected_logit)
        self.assertEqual(result.scores["none"].fake, expected_logit >= 0.5)
        self.assertIn(4, result.layer_margins["none"])
        self.assertAlmostEqual(result.duration_s, 0.1)

    def test_empty_audio_is_rejected(self):
        engine = Engine.__new__(Engine)
        engine.pc = FakePilotCommon()
        engine.cfg = {}
        with self.assertRaisesRegex(ValueError, "empty"):
            engine.analyse(np.array([], dtype=np.float32), 16000, "empty.wav")


if __name__ == "__main__":
    unittest.main()

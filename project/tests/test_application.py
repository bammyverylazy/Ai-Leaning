import argparse
import collections
import queue
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import config as C
from application import RecognitionWorker, build_parser
from subtitle import SubtitleManager


class RecognitionWorkerTests(unittest.TestCase):
    def test_worker_emits_speech_eligible_event_only_after_subtitle_commit(self):
        events = queue.Queue()
        manager = SubtitleManager(min_votes=2, idle_frames=1, cooldown_sec=0)
        worker = RecognitionWorker(
            mode="demo",
            camera_index=0,
            model=lambda _features: torch.tensor([[4.0, 0.0]]),
            vocab=["WHATFOR1", "CHRISTMAS1"],
            args=argparse.Namespace(),
            subtitle_manager=manager,
            events=events,
            model_version="test-model",
            session_id="test-session",
        )
        sequence = collections.deque(
            [np.zeros((C.N_POINTS, 3), dtype=np.float32) for _ in range(C.SEQ_LEN)],
            maxlen=C.SEQ_LEN,
        )
        landmarks = np.full((C.N_POINTS, 3), np.nan, dtype=np.float32)
        landmarks[C.LH.start, 0] = 0.5

        with (
            patch(
                "application.preprocess_sequence",
                return_value=(np.zeros((C.SEQ_LEN, C.N_POINTS, 3)), np.zeros((C.SEQ_LEN, 2))),
            ),
            patch(
                "application.build_features",
                return_value=np.zeros((C.SEQ_LEN, 590), dtype=np.float32),
            ),
        ):
            worker._recognize_live(sequence, landmarks)
            first_events = self._drain(events)
            self.assertFalse(any(name == "committed" for name, _ in first_events))

            worker._recognize_live(sequence, landmarks)
            committed_events = [
                value for name, value in self._drain(events) if name == "committed"
            ]
            self.assertEqual(len(committed_events), 1)
            self.assertEqual(committed_events[0][1], "WHAT FOR")
            self.assertEqual(manager.subtitle, "WHAT FOR")

            worker._recognize_live(sequence, landmarks)
            self.assertFalse(
                any(name == "committed" for name, _ in self._drain(events))
            )

    @staticmethod
    def _drain(events):
        collected = []
        while not events.empty():
            collected.append(events.get_nowait())
            events.task_done()
        return collected


class ApplicationArgumentsTests(unittest.TestCase):
    def test_parser_defaults_to_demo_and_accepts_testing_entry_mode(self):
        parser = build_parser()

        self.assertEqual(parser.parse_args([]).mode, "demo")
        self.assertEqual(
            parser.parse_args(["--mode", "testing", "--target", "AXE1"]).target,
            "AXE1",
        )
        self.assertTrue(parser.parse_args(["--debug-performance"]).debug_performance)


if __name__ == "__main__":
    unittest.main()

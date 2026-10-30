import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from subtitle import State, SubtitleManager


class SubtitleManagerTests(unittest.TestCase):
    def setUp(self):
        self.manager = SubtitleManager(min_votes=3, idle_frames=3, cooldown_sec=0)

    def confirm(self):
        confirmed = None
        for _ in range(3):
            confirmed, _ = self.manager.update(
                "WHAT FOR",
                0.9,
                True,
                original_model_label="WHATFOR1",
                class_index=49,
            )
        return confirmed

    def test_confirmed_subtitle_keeps_source_class_metadata(self):
        self.assertEqual(self.confirm(), "WHAT FOR")

        self.assertEqual(self.manager.word_list, ["WHAT FOR"])
        self.assertEqual(
            (self.manager.entries[0].original_model_label, self.manager.entries[0].class_index),
            ("WHATFOR1", 49),
        )

    def test_same_sign_requires_idle_before_it_can_be_confirmed_again(self):
        self.confirm()
        for _ in range(8):
            confirmed, state = self.manager.update("WHAT FOR", 0.9, True)
            self.assertIsNone(confirmed)
            self.assertEqual(state, State.COOLDOWN)
        self.assertEqual(self.manager.word_list, ["WHAT FOR"])

        for _ in range(self.manager.idle_frames):
            self.manager.update(None, 0.0, False)
        self.assertEqual(self.manager.state, State.IDLE)
        self.assertEqual(self.confirm(), "WHAT FOR")
        self.assertEqual(self.manager.word_list, ["WHAT FOR", "WHAT FOR"])

    def test_clear_only_clears_subtitle_state(self):
        self.confirm()
        self.manager.clear()

        self.assertEqual(self.manager.subtitle, "")
        self.assertEqual(self.manager.state, State.IDLE)
        self.assertEqual(self.manager.entries, [])

    def test_strong_consistent_prediction_uses_fast_confirmation(self):
        manager = SubtitleManager(min_votes=5, conf_threshold=0.45)

        self.assertIsNone(manager.update("HELLO", 0.96, True)[0])
        self.assertIsNone(manager.update("HELLO", 0.95, True)[0])
        confirmed, state = manager.update("HELLO", 0.94, True)

        self.assertEqual(confirmed, "HELLO")
        self.assertEqual(state, State.COOLDOWN)

    def test_normal_confidence_still_requires_full_stabilization(self):
        manager = SubtitleManager(min_votes=5, conf_threshold=0.45)

        for _ in range(4):
            confirmed, _ = manager.update("HELLO", 0.70, True)
            self.assertIsNone(confirmed)
        confirmed, _ = manager.update("HELLO", 0.70, True)

        self.assertEqual(confirmed, "HELLO")

    def test_unstable_high_confidence_prediction_does_not_fast_commit(self):
        manager = SubtitleManager(min_votes=5, conf_threshold=0.45)
        predictions = [
            ("HELLO", 0.96),
            ("THANK YOU", 0.95),
            ("HELLO", 0.94),
        ]

        results = [
            manager.update(label, confidence, True)[0]
            for label, confidence in predictions
        ]

        self.assertEqual(results, [None, None, None])
        self.assertEqual(manager.subtitle, "")


if __name__ == "__main__":
    unittest.main()

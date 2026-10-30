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


if __name__ == "__main__":
    unittest.main()

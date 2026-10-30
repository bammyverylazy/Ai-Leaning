import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sign_demo_catalog import SignDemo
from testing_mode import _play_demo


class TestingModePlaybackTests(unittest.TestCase):
    def setUp(self):
        self.demo = SignDemo(
            label="AXE1",
            video_id="axe-demo",
            version="1",
            path=Path("axe-v1.mp4"),
        )

    def test_demo_can_be_replayed_then_readied(self):
        capture = MagicMock()
        capture.isOpened.return_value = True
        capture.get.return_value = 25.0
        frames = np.zeros((48, 64, 3), dtype=np.uint8)
        capture.read.side_effect = [
            (True, frames.copy()),
            (False, None),
            (True, frames.copy()),
            (False, None),
        ]

        with (
            patch("testing_mode.cv2.VideoCapture", return_value=capture),
            patch("testing_mode.cv2.namedWindow"),
            patch("testing_mode.cv2.imshow"),
            patch("testing_mode.cv2.waitKey", side_effect=[ord("r"), 13]),
            patch("testing_mode.cv2.destroyWindow"),
        ):
            result = _play_demo("AXE1", self.demo)

        self.assertIsNotNone(result)
        self.assertTrue(result[0])
        self.assertGreaterEqual(result[1], 0)
        capture.set.assert_called_once()
        capture.release.assert_called_once()

    def test_unopenable_demo_fails_explicitly(self):
        capture = MagicMock()
        capture.isOpened.return_value = False

        with patch("testing_mode.cv2.VideoCapture", return_value=capture):
            with self.assertRaisesRegex(RuntimeError, "Could not open"):
                _play_demo("AXE1", self.demo)

        capture.release.assert_called_once()


if __name__ == "__main__":
    unittest.main()

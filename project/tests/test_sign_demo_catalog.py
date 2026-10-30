import csv
import tempfile
import unittest
from pathlib import Path
import sys
from unittest.mock import MagicMock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sign_demo_catalog import MANIFEST_FIELDS, load_catalog


class SignDemoCatalogTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.demo_root = self.root / "data" / "sign_demos"
        self.video = self.demo_root / "videos" / "AXE1" / "axe-v1.mp4"
        self.video.parent.mkdir(parents=True)
        self.video.write_bytes(b"test video")
        self.manifest = self.demo_root / "manifest.csv"
        self.demo_root.mkdir(parents=True, exist_ok=True)
        capture = MagicMock()
        capture.isOpened.return_value = True
        capture.read.return_value = (True, np.zeros((1, 1, 3), dtype=np.uint8))
        self.patches = (
            patch("sign_demo_catalog.C.ROOT", self.root),
            patch("sign_demo_catalog.C.SIGN_DEMOS", self.demo_root),
            patch("sign_demo_catalog.C.SIGN_DEMO_VIDEOS", self.demo_root / "videos"),
            patch("sign_demo_catalog.cv2.VideoCapture", return_value=capture),
        )
        for item in self.patches:
            item.start()

    def tearDown(self):
        for item in reversed(self.patches):
            item.stop()
        self.temp_dir.cleanup()

    def write_manifest(self, rows):
        with self.manifest.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=MANIFEST_FIELDS)
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def row(**overrides):
        result = {
            "label": "AXE1",
            "video_id": "axe-demo",
            "version": "1",
            "filename": "axe-v1.mp4",
            "source": "curator-reviewed source",
            "verification_status": "verified",
            "verified_meaning": "axe",
            "notes": "",
            "is_current": "true",
        }
        result.update(overrides)
        return result

    def test_only_verified_existing_mp4_is_available_for_exact_model_label(self):
        self.write_manifest([self.row()])

        catalog, missing = load_catalog(["AXE1"], self.manifest)

        self.assertEqual(catalog["AXE1"].path, self.video)
        self.assertEqual((catalog["AXE1"].video_id, catalog["AXE1"].version), ("axe-demo", "1"))
        self.assertEqual(missing, {})

    def test_unreviewed_demo_and_other_model_vocab_entries_are_not_available(self):
        self.write_manifest(
            [self.row(verification_status="unverified"), self.row(label="OTHER")]
        )

        catalog, missing = load_catalog(["AXE1"], self.manifest)

        self.assertEqual(catalog, {})
        self.assertIn("not verified", missing["AXE1"])

    def test_demo_filenames_cannot_escape_curated_demo_storage(self):
        self.write_manifest([self.row(filename="../../outside.mp4")])

        with self.assertRaises(ValueError):
            load_catalog(["AXE1"], self.manifest)

    def test_corrupted_video_is_not_added_to_available_catalog(self):
        self.write_manifest([self.row()])
        capture = MagicMock()
        capture.isOpened.return_value = True
        capture.read.return_value = (False, None)

        with patch("sign_demo_catalog.cv2.VideoCapture", return_value=capture):
            catalog, missing = load_catalog(["AXE1"], self.manifest)

        self.assertEqual(catalog, {})
        self.assertIn("no readable frames", missing["AXE1"])
        capture.release.assert_called_once()


if __name__ == "__main__":
    unittest.main()

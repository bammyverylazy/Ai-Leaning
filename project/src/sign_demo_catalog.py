"""Load curated, vocabulary-matched sign demonstration videos."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import cv2
import config as C

MANIFEST_FIELDS = (
    "label",
    "video_id",
    "version",
    "filename",
    "source",
    "verification_status",
    "verified_meaning",
    "notes",
    "is_current",
)


@dataclass(frozen=True)
class SignDemo:
    label: str
    video_id: str
    version: str
    path: Path


def load_catalog(
    vocab: list[str], manifest_path: Path = C.SIGN_DEMO_MANIFEST
) -> tuple[dict[str, SignDemo], dict[str, str]]:
    """Return verified available videos and a reason for every unavailable label."""
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Sign demo manifest not found: {manifest_path}")

    with manifest_path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None or not set(MANIFEST_FIELDS).issubset(reader.fieldnames):
            raise ValueError(
                f"Manifest must contain these columns: {', '.join(MANIFEST_FIELDS)}"
            )
        rows = list(reader)

    allowed_labels = set(vocab)
    current: dict[str, dict[str, str]] = {}
    for row in rows:
        label = (row.get("label") or "").strip()
        if not label:
            raise ValueError("Sign demo manifest contains a row with an empty label")
        if label not in allowed_labels:
            continue
        is_current = (row.get("is_current") or "").strip().lower()
        if is_current not in {"true", "false"}:
            raise ValueError(f"Invalid is_current value for {label}: {is_current!r}")
        if is_current == "true":
            if label in current:
                raise ValueError(f"More than one current demo is configured for {label}")
            current[label] = row

    catalog: dict[str, SignDemo] = {}
    missing: dict[str, str] = {}
    for label in vocab:
        row = current.get(label)
        if row is None:
            missing[label] = "no current demo entry"
            continue

        verification_status = (row.get("verification_status") or "").strip().lower()
        if verification_status != "verified":
            missing[label] = (
                "demo not verified "
                f"(status: {verification_status or 'empty'})"
            )
            continue

        video_id = (row.get("video_id") or "").strip()
        version = (row.get("version") or "").strip()
        filename = (row.get("filename") or "").strip()
        source = (row.get("source") or "").strip()
        verified_meaning = (row.get("verified_meaning") or "").strip()
        if not all((video_id, version, filename, source, verified_meaning)):
            raise ValueError(f"Verified demo for {label} has incomplete metadata")

        relative_path = Path(filename)
        if (
            relative_path.name != filename
            or relative_path.is_absolute()
            or relative_path.suffix.lower() != ".mp4"
        ):
            raise ValueError(f"Demo filename for {label} must be a simple .mp4 filename")
        path = (C.SIGN_DEMO_VIDEOS / label / relative_path).resolve()
        try:
            path.relative_to(C.SIGN_DEMO_VIDEOS.resolve())
        except ValueError as exc:
            raise ValueError(
                f"Demo path for {label} must be inside {C.SIGN_DEMO_VIDEOS}"
            ) from exc

        if not path.is_file():
            missing[label] = f"video file not found: {filename}"
            continue
        capture = cv2.VideoCapture(str(path))
        try:
            if not capture.isOpened():
                missing[label] = f"video cannot be opened: {filename}"
                continue
            readable, frame = capture.read()
            if not readable or frame is None or frame.size == 0:
                missing[label] = f"video contains no readable frames: {filename}"
                continue
        finally:
            capture.release()
        catalog[label] = SignDemo(label, video_id, version, path)

    return catalog, missing


def write_missing_report(
    missing: dict[str, str], report_path: Path = C.MISSING_SIGN_DEMOS_REPORT
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(("label", "status"))
        writer.writerows(missing.items())

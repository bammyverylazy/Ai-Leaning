"""validate_dataset.py — ตรวจสอบคุณภาพ dataset และสร้างรายงาน

ใช้งาน:
  python src/validate_dataset.py          # ตรวจ metadata + landmarks ถ้ามี
  python src/validate_dataset.py --quick  # ตรวจเฉพาะ metadata (เร็ว)

ผลลัพธ์:
  reports/dataset_report.json
  reports/dataset_report.txt
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import config as C


def check_signer_leakage(splits: dict[str, pd.DataFrame]) -> dict:
    issues = {}
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        if a in splits and b in splits:
            inter = set(splits[a].participant) & set(splits[b].participant)
            issues[f"{a}/{b}_overlap"] = sorted(inter)
    return issues


def check_landmarks(df: pd.DataFrame, split: str) -> dict:
    missing, too_short, low_hand = [], [], []
    for r in df.itertuples():
        lm_path = C.LANDMARKS / (r.video.rsplit(".", 1)[0] + ".npy")
        if not lm_path.exists():
            missing.append(r.video)
            continue
        arr = np.load(lm_path)
        if arr.shape[0] < 8:
            too_short.append(r.video)
            continue
        hand_ratio = float(
            (~np.isnan(arr[:, C.LH, 0]).any(axis=1) | ~np.isnan(arr[:, C.RH, 0]).any(axis=1)).mean()
        )
        if hand_ratio < 0.2:
            low_hand.append((r.video, round(hand_ratio, 3)))
    return {
        "missing_landmarks": len(missing),
        "too_short": len(too_short),
        "low_hand_ratio": len(low_hand),
        "missing_examples": missing[:10],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="ตรวจเฉพาะ metadata")
    args = ap.parse_args()

    report: dict = {}
    t0 = time.time()

    # ---- Raw metadata ----
    raw_stats = {}
    for split in ["train", "val", "test"]:
        try:
            path = C.find_split_csv(split)
            df_raw = pd.read_csv(path)
            raw_stats[split] = {
                "rows": len(df_raw),
                "columns": list(df_raw.columns),
                "glosses": int(df_raw["Gloss"].nunique()) if "Gloss" in df_raw.columns else None,
                "signers": int(df_raw["Participant ID"].nunique()) if "Participant ID" in df_raw.columns else None,
            }
        except FileNotFoundError as e:
            raw_stats[split] = {"error": str(e)}
    report["raw_metadata"] = raw_stats

    # ---- Subset (ถ้ามี) ----
    splits: dict[str, pd.DataFrame] = {}
    vocab_words = []
    subset_exists = (C.DATA / "vocab.txt").exists()
    if subset_exists:
        vocab_words = [l.strip() for l in (C.DATA / "vocab.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
        for split in ["train", "val", "test"]:
            p = C.DATA / f"subset_{split}.csv"
            if p.exists():
                splits[split] = pd.read_csv(p)

        subset_stats = {}
        for s, df in splits.items():
            subset_stats[s] = {
                "samples": len(df),
                "signers": int(df.participant.nunique()),
                "glosses": int(df.gloss.nunique()),
            }
        report["subset"] = {
            "vocab_size": len(vocab_words),
            "vocab": vocab_words,
            "splits": subset_stats,
        }

        # leakage check
        report["signer_leakage"] = check_signer_leakage(splits)

        # class distribution
        if "train" in splits:
            vc = splits["train"].gloss.value_counts()
            report["class_distribution"] = {
                "min": int(vc.min()),
                "max": int(vc.max()),
                "mean": round(float(vc.mean()), 1),
                "median": round(float(vc.median()), 1),
            }

        # landmark check (ถ้าไม่ quick)
        if not args.quick and C.LANDMARKS.exists():
            lm_report = {}
            for s, df in splits.items():
                print(f"ตรวจ landmarks สำหรับ {s} ({len(df)} ตัวอย่าง)...")
                lm_report[s] = check_landmarks(df, s)
            report["landmarks"] = lm_report
    else:
        report["subset"] = "ยังไม่ได้รัน make_vocab.py"

    # ---- Features (ถ้ามี) ----
    feat_stats = {}
    for s in ["train", "val", "test"]:
        fp = C.FEATURES / f"{s}.npz"
        if fp.exists():
            d = np.load(fp, allow_pickle=True)
            feat_stats[s] = {
                "samples": int(d["y"].shape[0]),
                "pos_shape": list(d["pos"].shape),
                "classes": int(np.unique(d["y"]).shape[0]),
            }
    report["features"] = feat_stats if feat_stats else "ยังไม่ได้รัน dataset.py"

    # ---- Model ----
    model_files = list(C.MODELS.glob("*.pt")) if C.MODELS.exists() else []
    report["models"] = [str(p.name) for p in model_files]

    report["elapsed_sec"] = round(time.time() - t0, 1)

    # บันทึก
    C.REPORTS.mkdir(exist_ok=True)
    (C.REPORTS / "dataset_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    # สรุปข้อความ
    lines = ["=" * 60, "DATASET REPORT", "=" * 60]
    lines.append("\n[Raw ASL Citizen metadata]")
    for s, d in report["raw_metadata"].items():
        if "error" not in d:
            lines.append(f"  {s}: {d['rows']} rows, {d['signers']} signers, {d['glosses']} glosses")
        else:
            lines.append(f"  {s}: ERROR - {d['error']}")

    if isinstance(report.get("subset"), dict):
        lines.append(f"\n[Subset] vocab={report['subset']['vocab_size']} words")
        for s, d in report["subset"]["splits"].items():
            lines.append(f"  {s}: {d['samples']} samples, {d['signers']} signers")

    leak = report.get("signer_leakage", {})
    any_leak = any(v for v in leak.values())
    lines.append("\n[Signer leakage]")
    for k, v in leak.items():
        status = "⚠ LEAK" if v else "OK"
        lines.append(f"  {k}: {status} {v if v else ''}")

    if isinstance(report.get("features"), dict):
        lines.append("\n[Extracted features]")
        for s, d in report["features"].items():
            lines.append(f"  {s}: {d['samples']} samples, shape {d['pos_shape']}")

    lines.append(f"\n[Models]: {report['models']}")
    lines.append(f"\nTime: {report['elapsed_sec']}s")
    lines.append("=" * 60)

    txt = "\n".join(lines)
    print(txt)
    (C.REPORTS / "dataset_report.txt").write_text(txt, encoding="utf-8")
    print(f"\nรายงานบันทึกที่: {C.REPORTS / 'dataset_report.json'}")


if __name__ == "__main__":
    main()

"""record_self.py — เก็บวิดีโอตัวเองเพื่อเพิ่ม training data

วิธีใช้:
  python src/record_self.py --word HELLO --signer myname --reps 5
  python src/record_self.py --word HELLO --signer myname --reps 5 --split train

ปุ่ม:
  SPACE = เริ่ม/หยุดบันทึก
  R     = ทำซ้ำ clip ล่าสุด (ลบแล้วบันทึกใหม่)
  Q/ESC = ออก

ผลลัพธ์:
  data/raw/self_recorded/<WORD>/<signer>_<session>_<n>.mp4
  data/metadata/self_recorded_manifest.csv   (append)
"""
from __future__ import annotations

import argparse
import csv
import os
import time
import uuid
from pathlib import Path

import cv2
import numpy as np

import config as C

SELF_DIR = C.RAW / "self_recorded"
MANIFEST = C.DATA / "metadata" / "self_recorded_manifest.csv"
MANIFEST_FIELDS = [
    "video_id", "gloss", "signer_id", "session_id",
    "recording_date", "source", "split", "video_path",
]


def draw_text(img, text, org, scale=0.7, color=(255, 255, 255)):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def record_clips(word: str, signer: str, reps: int, split: str, camera: int):
    word_dir = SELF_DIR / word
    word_dir.mkdir(parents=True, exist_ok=True)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)

    # session id สำหรับกลุ่มการบันทึกนี้
    session_id = uuid.uuid4().hex[:8]

    # เปิดไฟล์ manifest (append)
    manifest_new = not MANIFEST.exists()
    mf = open(MANIFEST, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(mf, fieldnames=MANIFEST_FIELDS)
    if manifest_new:
        writer.writeheader()

    cap = cv2.VideoCapture(camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        raise SystemExit(f"เปิดกล้อง {camera} ไม่ได้")

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    completed = 0
    recording = False
    buf_frames = []
    last_out = None

    print(f"\nบันทึกคำ: {word}  |  ผู้แสดง: {signer}  |  เป้าหมาย: {reps} คลิป  |  split: {split}")
    print("SPACE=start/stop  R=redo  Q=quit\n")

    while completed < reps:
        ok, frame = cap.read()
        if not ok:
            break
        display = cv2.flip(frame, 1)

        # Header
        prog = f"[{completed}/{reps}]  WORD: {word}"
        draw_text(display, prog, (10, 30), 0.8, (0, 220, 220))

        if recording:
            buf_frames.append(frame.copy())
            # แสดง REC badge
            cv2.circle(display, (W - 30, 30), 12, (0, 0, 220), -1)
            draw_text(display, f"REC  {len(buf_frames)/fps:.1f}s", (W - 130, 35), 0.6, (0, 0, 220))
        else:
            if completed == 0:
                draw_text(display, "กด SPACE เริ่มบันทึก", (10, H // 2), 0.9, (255, 255, 255))
            else:
                draw_text(display, f"ทำครั้งที่ {completed+1} — SPACE เริ่ม", (10, H // 2), 0.8, (200, 200, 200))
            if last_out:
                draw_text(display, f"บันทึกแล้ว: {last_out.name}", (10, H - 40), 0.55, (50, 220, 50))
                draw_text(display, "R=redo ถ้าไม่พอใจ", (10, H - 20), 0.5, (180, 180, 180))

        draw_text(display, "Q=quit", (10, H - 10), 0.5, (150, 150, 150))
        cv2.imshow(f"Record: {word}", display)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break

        elif key == ord(" "):
            if not recording:
                buf_frames = []
                recording = True
            else:
                recording = False
                if len(buf_frames) < int(fps * 0.5):  # สั้นเกิน 0.5 วินาที
                    print("คลิปสั้นเกิน ลองใหม่")
                    buf_frames = []
                    continue
                # บันทึกไฟล์
                n = completed + 1
                out_name = f"{signer}_{session_id}_{n:02d}.mp4"
                out_path = word_dir / out_name
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                vw = cv2.VideoWriter(str(out_path), fourcc, fps, (W, H))
                for f in buf_frames:
                    vw.write(f)
                vw.release()
                last_out = out_path
                print(f"  ✓ บันทึก {out_path.name}  ({len(buf_frames)} เฟรม, {len(buf_frames)/fps:.1f}s)")
                # เขียน manifest
                video_id = f"self_{signer}_{session_id}_{n:02d}"
                rel_path = str(out_path.relative_to(C.ROOT))
                writer.writerow({
                    "video_id": video_id,
                    "gloss": word,
                    "signer_id": signer,
                    "session_id": session_id,
                    "recording_date": time.strftime("%Y-%m-%d"),
                    "source": "self_recorded",
                    "split": split,
                    "video_path": rel_path,
                })
                mf.flush()
                completed += 1

        elif key == ord("r") and last_out and not recording:
            # ลบ clip ล่าสุด
            last_out.unlink(missing_ok=True)
            completed -= 1
            print(f"  ✗ ลบ {last_out.name} แล้ว")
            last_out = None

    cap.release()
    cv2.destroyAllWindows()
    mf.close()
    print(f"\nเสร็จ: บันทึก {completed} คลิปของคำ '{word}'")
    print(f"manifest อยู่ที่: {MANIFEST}")


def main():
    ap = argparse.ArgumentParser(description="บันทึกวิดีโอ ASL สำหรับเพิ่ม training data")
    ap.add_argument("--word", required=True, help="คำ ASL ที่ต้องการบันทึก เช่น HELLO")
    ap.add_argument("--signer", required=True, help="ชื่อผู้แสดง (ใช้ตัวอักษรและตัวเลขเท่านั้น)")
    ap.add_argument("--reps", type=int, default=5, help="จำนวน clip ที่ต้องการ")
    ap.add_argument("--split", choices=["train", "val", "test"], default="train",
                    help="กำหนด split (ใช้ signer คนละกลุ่มกับ test เสมอ)")
    ap.add_argument("--camera", type=int, default=0)
    args = ap.parse_args()

    signer = "".join(c for c in args.signer if c.isalnum() or c in "_-")
    if not signer:
        raise SystemExit("--signer ต้องมีตัวอักษรหรือตัวเลข")
    word = args.word.upper()
    record_clips(word, signer, args.reps, args.split, args.camera)


if __name__ == "__main__":
    main()

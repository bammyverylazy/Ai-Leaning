"""subtitle.py — จัดการ subtitle แบบ word-by-word พร้อม state machine สำหรับตรวจจับขอบเขตสัญญาณ

State machine:
  IDLE → CANDIDATE → CONFIRMED → COOLDOWN → IDLE

- IDLE:      ไม่มีสัญญาณมือ หรือ confidence ต่ำ
- CANDIDATE: เห็นคำเดิมซ้ำ ๆ เกิน min_votes ครั้ง รอ confirm
- CONFIRMED: ยืนยันคำแล้ว เพิ่มใน subtitle
- COOLDOWN:  รอให้ผู้แสดงหยุดพัก ก่อนจะ re-arm รับคำใหม่
"""
from __future__ import annotations

import collections
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Tuple


class State(Enum):
    IDLE = auto()
    CANDIDATE = auto()
    CONFIRMED = auto()
    COOLDOWN = auto()


@dataclass(frozen=True)
class SubtitleEntry:
    canonical_label: str
    original_model_label: str
    class_index: Optional[int]


@dataclass
class SubtitleManager:
    # ---- Tuning params ----
    min_votes: int = 5           # ต้องเห็นคำเดียวกันอย่างน้อยกี่ครั้งต่อเนื่อง จึง CONFIRM
    vote_window: int = 10        # ขนาด sliding window สำหรับนับ votes
    conf_threshold: float = 0.45 # confidence ต่ำกว่านี้ = ไม่นับ
    cooldown_sec: float = 0.8    # minimum time before the subtitle manager can re-arm
    idle_frames: int = 8         # consecutive invalid updates required to re-arm
    max_subtitle_words: int = 50 # จำกัดความยาว subtitle

    # ---- Internal state ----
    _state: State = field(default=State.IDLE, init=False)
    _votes: collections.deque = field(default_factory=lambda: collections.deque(), init=False)
    _idle_count: int = field(default=0, init=False)
    _confirmed_word: str = field(default="", init=False)
    _confirmed_conf: float = field(default=0.0, init=False)
    _cooldown_start: float = field(default=0.0, init=False)
    _subtitle_entries: List[SubtitleEntry] = field(default_factory=list, init=False)

    def update(
        self,
        word: Optional[str],
        conf: float,
        hands_present: bool,
        *,
        original_model_label: Optional[str] = None,
        class_index: Optional[int] = None,
    ) -> Tuple[Optional[str], State]:
        """เรียกทุกครั้งที่มี prediction ใหม่
        
        Args:
            word:          คำที่โมเดลทำนาย (None = ไม่มี prediction)
            conf:          confidence score (0–1)
            hands_present: True ถ้า MediaPipe เจอมือในเฟรมนี้

        Returns:
            (newly_confirmed_word, current_state)
            newly_confirmed_word = ชื่อคำที่เพิ่ง CONFIRM (None ถ้าไม่มี)
        """
        newly_confirmed: Optional[str] = None
        valid = hands_present and word is not None and conf >= self.conf_threshold

        # ---- COOLDOWN ----
        if self._state == State.COOLDOWN:
            if not valid:
                self._idle_count += 1
            else:
                self._idle_count = 0
            if (
                time.monotonic() - self._cooldown_start >= self.cooldown_sec
                and self._idle_count >= self.idle_frames
            ):
                self._state = State.IDLE
                self._votes.clear()
                self._idle_count = 0
            return newly_confirmed, self._state

        # ---- ถ้าไม่มี valid prediction ----
        if not valid:
            self._idle_count += 1
            if self._state == State.CANDIDATE and self._idle_count >= self.idle_frames:
                # คำหลุดก่อน confirm → กลับ IDLE
                self._state = State.IDLE
                self._votes.clear()
                self._idle_count = 0
            return newly_confirmed, self._state

        # มี valid prediction
        self._idle_count = 0
        self._votes.append((word, conf))
        if len(self._votes) > self.vote_window:
            self._votes.popleft()

        # นับ majority vote ใน window
        counter: dict[str, float] = {}
        for w, _ in self._votes:
            counter[w] = counter.get(w, 0) + 1
        best_word = max(counter, key=lambda k: counter[k])
        best_count = counter[best_word]

        # ---- IDLE / CANDIDATE ----
        if self._state == State.IDLE:
            if best_count >= 2:
                self._state = State.CANDIDATE

        if self._state == State.CANDIDATE:
            if best_word != (self._votes[-2][0] if len(self._votes) >= 2 else best_word):
                # คำเปลี่ยน → reset candidate
                self._votes.clear()
                self._votes.append((word, conf))
                self._state = State.IDLE
            elif best_count >= self.min_votes:
                # ยืนยัน!
                avg_conf = sum(c for w, c in self._votes if w == best_word) / best_count
                self._confirmed_word = best_word
                self._confirmed_conf = avg_conf
                self._add_word(
                    best_word,
                    original_model_label or best_word,
                    class_index,
                )
                newly_confirmed = best_word
                self._state = State.CONFIRMED
                self._cooldown_start = time.monotonic()
                self._votes.clear()
                # ย้ายไป COOLDOWN ทันที
                self._state = State.COOLDOWN

        return newly_confirmed, self._state

    def _add_word(
        self, word: str, original_model_label: str, class_index: Optional[int]
    ) -> None:
        self._subtitle_entries.append(
            SubtitleEntry(word, original_model_label, class_index)
        )
        if len(self._subtitle_entries) > self.max_subtitle_words:
            self._subtitle_entries.pop(0)

    # ---- Public accessors ----
    @property
    def subtitle(self) -> str:
        return " ".join(entry.canonical_label for entry in self._subtitle_entries)

    @property
    def word_list(self) -> List[str]:
        return [entry.canonical_label for entry in self._subtitle_entries]

    @property
    def entries(self) -> List[SubtitleEntry]:
        return list(self._subtitle_entries)

    @property
    def idle_count(self) -> int:
        return self._idle_count

    @property
    def state(self) -> State:
        return self._state

    @property
    def candidate_word(self) -> str:
        """คำที่กำลัง candidate อยู่ (ถ้ามี)"""
        if not self._votes:
            return ""
        counter: dict[str, int] = {}
        for w, _ in self._votes:
            counter[w] = counter.get(w, 0) + 1
        return max(counter, key=lambda k: counter[k])

    @property
    def candidate_progress(self) -> float:
        """0–1 แสดงความคืบหน้าไปสู่ confirmation"""
        if not self._votes:
            return 0.0
        counter: dict[str, int] = {}
        for w, _ in self._votes:
            counter[w] = counter.get(w, 0) + 1
        best_count = max(counter.values())
        return min(best_count / self.min_votes, 1.0)

    def clear(self) -> None:
        """ล้าง subtitle และ reset state"""
        self._subtitle_entries.clear()
        self._votes.clear()
        self._state = State.IDLE
        self._idle_count = 0
        self._confirmed_word = ""

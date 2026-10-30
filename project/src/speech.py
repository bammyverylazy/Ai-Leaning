"""Offline, non-blocking speech for newly committed subtitle entries."""
from __future__ import annotations

import base64
import queue
import re
import shutil
import subprocess
import threading
from dataclasses import dataclass
from typing import Callable


def spoken_text(canonical_label: str) -> str:
    """Turn a displayed label into a concise spoken English phrase."""
    text = re.sub(r"\s+\(variant\s+\d+\)$", "", canonical_label, flags=re.IGNORECASE)
    return text.replace("/", " or ").replace("_", " ").title().replace(" Or ", " or ")


@dataclass(frozen=True)
class _SpeechItem:
    generation: int
    text: str


class SpeechService:
    """Serialize local SAPI speech and make mute/clear cancel queued audio."""

    def __init__(
        self,
        on_error: Callable[[str], None],
        *,
        powershell: str | None = None,
        speaker: Callable[[str, threading.Event], None] | None = None,
    ) -> None:
        self._powershell = powershell or shutil.which("powershell.exe")
        self._speaker = speaker
        self._on_error = on_error
        self._queue: queue.Queue[_SpeechItem | None] = queue.Queue()
        self._lock = threading.Lock()
        self._enabled = True
        self._closed = False
        self._generation = 0
        self._active_cancel: threading.Event | None = None
        self._thread = threading.Thread(
            target=self._run, name="asl-speech", daemon=True
        )
        self._thread.start()

    @property
    def available(self) -> bool:
        return self._speaker is not None or self._powershell is not None

    @property
    def enabled(self) -> bool:
        with self._lock:
            return self._enabled

    def set_enabled(self, enabled: bool) -> None:
        with self._lock:
            self._enabled = enabled
            if not enabled:
                self._cancel_locked()

    def enqueue(self, canonical_label: str) -> bool:
        text = spoken_text(canonical_label)
        if not text:
            return False
        with self._lock:
            if not self._enabled or self._closed:
                return False
            self._queue.put(_SpeechItem(self._generation, text))
        return True

    def cancel_pending(self) -> None:
        with self._lock:
            self._cancel_locked()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._enabled = False
            self._cancel_locked()
            self._queue.put(None)
        self._thread.join(timeout=2)

    def _cancel_locked(self) -> None:
        self._generation += 1
        if self._active_cancel is not None:
            self._active_cancel.set()
        while True:
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except queue.Empty:
                break

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    return
                with self._lock:
                    if (
                        self._closed
                        or not self._enabled
                        or item.generation != self._generation
                    ):
                        continue
                    cancel = threading.Event()
                    self._active_cancel = cancel
                try:
                    if self._speaker is not None:
                        self._speaker(item.text, cancel)
                    else:
                        self._speak(item.text, cancel)
                except Exception as exc:
                    self._on_error(str(exc))
                finally:
                    with self._lock:
                        if self._active_cancel is cancel:
                            self._active_cancel = None
            finally:
                self._queue.task_done()

    def _speak(self, text: str, cancel: threading.Event) -> None:
        if self._powershell is None:
            raise RuntimeError(
                "Offline Windows speech is unavailable; PowerShell SAPI was not found"
            )
        text_base64 = base64.b64encode(text.encode("utf-16le")).decode("ascii")
        script = (
            "$ErrorActionPreference='Stop'; "
            "Add-Type -AssemblyName System.Speech; "
            "$s=New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            "$text=[Text.Encoding]::Unicode.GetString("
            f"[Convert]::FromBase64String('{text_base64}')); "
            "$voice=$s.GetInstalledVoices() | Where-Object { "
            "$_.Enabled -and $_.VoiceInfo.Culture.Name -like 'en-*' "
            "} | Select-Object -First 1; "
            "if(!$voice){throw 'No enabled English Windows speech voice is installed'}; "
            "$s.SelectVoice($voice.VoiceInfo.Name); "
            "$s.Speak($text); $s.Dispose()"
        )
        encoded_script = base64.b64encode(script.encode("utf-16le")).decode("ascii")
        creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        process = subprocess.Popen(
            [
                self._powershell,
                "-NoLogo",
                "-NoProfile",
                "-NonInteractive",
                "-EncodedCommand",
                encoded_script,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            creationflags=creation_flags,
        )
        try:
            while process.poll() is None:
                if cancel.wait(0.05):
                    process.terminate()
                    try:
                        process.wait(timeout=0.5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=0.5)
                    return
            stderr = process.stderr.read() if process.stderr is not None else b""
            if process.returncode:
                message = stderr.decode(errors="replace").strip()
                raise RuntimeError(message or "Windows SAPI speech failed")
        finally:
            if process.stderr is not None:
                process.stderr.close()

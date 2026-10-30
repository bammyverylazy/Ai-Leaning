"""Offline, non-blocking speech for newly committed subtitle entries."""
from __future__ import annotations

import base64
import itertools
import queue
import re
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import TextIO, Callable


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
        on_started: Callable[[str, float], None] | None = None,
        on_ready: Callable[[float], None] | None = None,
    ) -> None:
        self._powershell = powershell or shutil.which("powershell.exe")
        self._speaker = speaker
        self._on_error = on_error
        self._on_started = on_started
        self._on_ready = on_ready
        self._queue: queue.Queue[_SpeechItem | None] = queue.Queue()
        self._sapi_events: queue.Queue[str] = queue.Queue()
        self._lock = threading.Lock()
        self._enabled = True
        self._closed = False
        self._generation = 0
        self._active_cancel: threading.Event | None = None
        self._process: subprocess.Popen[str] | None = None
        self._output_reader: threading.Thread | None = None
        self._item_ids = itertools.count(1)
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
        if self._speaker is None and self._powershell is not None:
            try:
                self._start_sapi_host()
            except Exception as exc:
                self._on_error(str(exc))
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    self._stop_sapi_host()
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
                        if self._on_started is not None:
                            self._on_started(item.text, time.perf_counter())
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
        if self._process is None or self._process.poll() is not None:
            self._start_sapi_host()
        process = self._process
        if process is None or process.stdin is None:
            raise RuntimeError("The offline speech process did not start.")
        item_id = next(self._item_ids)
        text_base64 = base64.b64encode(text.encode("utf-8")).decode("ascii")
        process.stdin.write(f"SPEAK|{item_id}|{text_base64}\n")
        process.stdin.flush()
        started = False
        while not cancel.is_set():
            try:
                message = self._sapi_events.get(timeout=0.05)
            except queue.Empty:
                if process.poll() is not None:
                    self._stop_sapi_host()
                    raise RuntimeError("The offline Windows speech process stopped.")
                continue
            kind, separator, message_id = message.partition("|")
            if not separator:
                if kind.startswith("ERROR:"):
                    self._stop_sapi_host()
                    raise RuntimeError(kind.removeprefix("ERROR:").strip())
                continue
            if message_id != str(item_id):
                continue
            if kind == "START" and not started:
                started = True
                if self._on_started is not None:
                    self._on_started(text, time.perf_counter())
            elif kind == "DONE":
                return
        self._stop_sapi_host()

    def _start_sapi_host(self) -> None:
        if self._process is not None and self._process.poll() is None:
            return
        if self._powershell is None:
            raise RuntimeError(
                "Offline Windows speech is unavailable; PowerShell SAPI was not found"
            )
        self._drain_sapi_events()
        script = (
            "$ErrorActionPreference='Stop'; "
            "try { "
            "Add-Type -AssemblyName System.Speech; "
            "$s=New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            "$voice=$s.GetInstalledVoices() | Where-Object { "
            "$_.Enabled -and $_.VoiceInfo.Culture.Name -like 'en-*' "
            "} | Select-Object -First 1; "
            "if(!$voice){throw 'No enabled English Windows speech voice is installed'}; "
            "$s.SelectVoice($voice.VoiceInfo.Name); "
            "[Console]::Out.WriteLine('READY'); [Console]::Out.Flush(); "
            "while($null -ne ($line=[Console]::In.ReadLine())) { "
            "$parts=$line -split '\\|',3; "
            "if($parts[0] -ne 'SPEAK' -or $parts.Length -ne 3){continue}; "
            "$id=$parts[1]; "
            "$text=[Text.Encoding]::UTF8.GetString("
            "[Convert]::FromBase64String($parts[2])); "
            "[Console]::Out.WriteLine('START|' + $id); [Console]::Out.Flush(); "
            "$s.Speak($text); "
            "[Console]::Out.WriteLine('DONE|' + $id); [Console]::Out.Flush() "
            "}; $s.Dispose() "
            "} catch { "
            "[Console]::Out.WriteLine('ERROR: ' + $_.Exception.Message); "
            "[Console]::Out.Flush(); exit 1 "
            "}"
        )
        encoded_script = base64.b64encode(script.encode("utf-16le")).decode("ascii")
        creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        started_at = time.monotonic()
        process = subprocess.Popen(
            [
                self._powershell,
                "-NoLogo",
                "-NoProfile",
                "-NonInteractive",
                "-EncodedCommand",
                encoded_script,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            bufsize=1,
            creationflags=creation_flags,
        )
        self._process = process
        if process.stdout is not None:
            self._output_reader = threading.Thread(
                target=self._read_sapi_events,
                args=(process.stdout,),
                name="asl-speech-events",
                daemon=True,
            )
            self._output_reader.start()
        while process.poll() is None:
            if self._closed:
                self._stop_sapi_host()
                return
            if time.monotonic() - started_at > 15:
                self._stop_sapi_host()
                raise RuntimeError("Timed out initializing Windows offline speech.")
            try:
                message = self._sapi_events.get(timeout=0.05)
            except queue.Empty:
                continue
            if message == "READY":
                if self._on_ready is not None:
                    self._on_ready((time.monotonic() - started_at) * 1000)
                return
            if message.startswith("ERROR:"):
                self._stop_sapi_host()
                raise RuntimeError(message.removeprefix("ERROR:").strip())
        self._stop_sapi_host()
        raise RuntimeError("The offline Windows speech process exited during startup.")

    def _read_sapi_events(self, output: TextIO) -> None:
        for line in output:
            self._sapi_events.put(line.strip())

    def _stop_sapi_host(self) -> None:
        process = self._process
        self._process = None
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=0.5)
        if self._output_reader is not None:
            self._output_reader.join(timeout=0.5)
            self._output_reader = None
        if process is not None:
            if process.stdin is not None:
                process.stdin.close()
            if process.stdout is not None:
                process.stdout.close()
        self._drain_sapi_events()

    def _drain_sapi_events(self) -> None:
        while True:
            try:
                self._sapi_events.get_nowait()
            except queue.Empty:
                return

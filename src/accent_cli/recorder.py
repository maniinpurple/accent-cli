"""Microphone recording -> 16 kHz mono 16-bit WAV.

Press Enter to start, Enter (or Ctrl+C) to stop. stdin is polled instead of read
from a background thread, so no stray input() is left waiting to swallow the
next menu keypress.
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.live import Live
from rich.text import Text

SAMPLE_RATE = 16_000


class RecorderError(RuntimeError):
    pass


def default_recording_path() -> Path:
    folder = Path(tempfile.gettempdir()) / "accent-cli"
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"rec_{datetime.now():%Y%m%d_%H%M%S}.wav"


def _enter_pressed() -> bool:
    """Non-blocking check for an Enter keypress (POSIX select / Windows msvcrt)."""
    if os.name == "nt":
        import msvcrt

        while msvcrt.kbhit():
            if msvcrt.getwch() in ("\r", "\n"):
                return True
        return False
    import select

    ready, _, _ = select.select([sys.stdin], [], [], 0)
    if ready:
        sys.stdin.readline()
        return True
    return False


def _flush_stdin() -> None:
    try:
        if os.name == "nt":
            import msvcrt

            while msvcrt.kbhit():
                msvcrt.getwch()
        elif sys.stdin.isatty():
            import termios

            termios.tcflush(sys.stdin, termios.TCIFLUSH)
    except Exception:  # noqa: BLE001 — best effort
        pass


def _meter(elapsed: float, rms: float, target_text: str | None = None) -> Text:
    db = 20 * np.log10(max(rms, 1e-6))
    filled = int(np.clip((db + 60) / 60, 0, 1) * 30)
    text = Text()
    if target_text:
        text.append("Say: ", style="dim")
        text.append(f"{target_text}\n", style="bold")
    text.append("● REC ", style="bold red")
    text.append(f"{elapsed:5.1f}s  ")
    text.append("▮" * filled, style="green" if filled < 26 else "red")
    text.append("▯" * (30 - filled), style="dim")
    text.append("   Enter/Ctrl+C to stop", style="dim")
    return text


def save_wav(audio: np.ndarray, path: Path, samplerate: int = SAMPLE_RATE) -> Path:
    from scipy.io import wavfile

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = (np.clip(audio.reshape(-1), -1.0, 1.0) * 32767).astype(np.int16)
    wavfile.write(str(path), samplerate, pcm)
    return path


def record_to_wav(
    out_path: Path | None = None,
    samplerate: int = SAMPLE_RATE,
    console: Console | None = None,
    target_text: str | None = None,
    wait_for_enter: bool = True,
) -> Path:
    console = console or Console()
    try:
        import sounddevice as sd
    except OSError as exc:  # PortAudio missing
        raise RecorderError(f"sounddevice/PortAudio unavailable: {exc}") from exc

    out_path = Path(out_path) if out_path else default_recording_path()
    chunks: list[np.ndarray] = []
    level = [0.0]

    def callback(indata, frames, time_info, status):  # noqa: ARG001
        chunks.append(indata.copy())
        level[0] = float(np.sqrt(np.mean(indata**2)))

    if wait_for_enter:
        console.input("[bold]Press Enter[/] to start recording… ")
    try:
        stream = sd.InputStream(samplerate=samplerate, channels=1, dtype="float32", callback=callback)
    except Exception as exc:  # noqa: BLE001
        raise RecorderError(
            f"Could not open microphone ({exc}). On macOS, allow mic access for your terminal in "
            "System Settings → Privacy & Security → Microphone."
        ) from exc

    _flush_stdin()  # a double-tapped Enter must not stop the take instantly
    start = time.monotonic()
    with stream:
        try:
            with Live(_meter(0, 0, target_text), console=console, transient=True, refresh_per_second=15) as live:
                while not _enter_pressed():
                    live.update(_meter(time.monotonic() - start, level[0], target_text))
                    time.sleep(0.05)
        except KeyboardInterrupt:
            pass

    audio = np.concatenate(chunks) if chunks else np.zeros((0, 1), dtype=np.float32)
    seconds = len(audio) / samplerate
    if seconds < 0.5:
        raise RecorderError("Recording too short (< 0.5 s).")
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    if peak < 0.01:
        console.print("[yellow]Warning: the recording is almost silent — check your mic input.[/]")
    save_wav(audio, out_path, samplerate)
    console.print(f"[dim]Saved {seconds:.1f}s recording → {out_path}[/]")
    return out_path

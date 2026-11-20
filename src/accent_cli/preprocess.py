"""Audio clean-up before ASR + prosody: high-pass → trim silence → peak normalize.

Trimming keeps leading/trailing silence from leaking into the first/last word's
timestamps (which skews nPVI and the function/content ratio). All word
timestamps downstream are relative to the *trimmed* audio; `trimmed_lead_ms`
in the returned info maps them back to the original file.
"""

from __future__ import annotations

from typing import Any

import numpy as np

SAMPLE_RATE = 16_000

HIGHPASS_HZ = 80.0          # below the lowest F0 we track (75 Hz floor), removes fan/AC rumble
TRIM_PAD_MS = 50            # keep a little air so initial plosive bursts (/p t k/) survive
TARGET_PEAK_DBFS = -3.0
FRAME_MS = 20
HOP_MS = 10
MIN_ACTIVE_MS = 80          # sustained energy needed to count as speech; a 5 ms key click
                            # smears to ~50 ms across overlapping frames, real words are longer


def highpass(audio: np.ndarray, sr: int = SAMPLE_RATE, cutoff: float = HIGHPASS_HZ, order: int = 4) -> np.ndarray:
    from scipy.signal import butter, sosfiltfilt

    if audio.size < sr // 10:  # too short for zero-phase filtering
        return audio.astype(np.float32)
    sos = butter(order, cutoff, btype="highpass", fs=sr, output="sos")
    return sosfiltfilt(sos, audio.astype(np.float64)).astype(np.float32)


def frame_db(audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    frame, hop = int(sr * FRAME_MS / 1000), int(sr * HOP_MS / 1000)
    if audio.size < frame:
        return np.zeros(0)
    n = 1 + (audio.size - frame) // hop
    idx = np.arange(frame)[None, :] + hop * np.arange(n)[:, None]
    rms = np.sqrt(np.mean(audio[idx].astype(np.float64) ** 2, axis=1))
    return 20 * np.log10(np.maximum(rms, 1e-8))


def speech_bounds(audio: np.ndarray, sr: int = SAMPLE_RATE, pad_ms: int = TRIM_PAD_MS) -> tuple[int, int]:
    """Sample indices [start, end) of the speech region, padded. Whole clip if nothing stands out."""
    db = frame_db(audio, sr)
    if db.size == 0:
        return 0, audio.size
    ref = np.percentile(db, 95)        # robust "loud speech" level
    noise = np.percentile(db, 10)      # robust noise floor
    threshold = min(ref - 10, max(ref - 30, noise + 6, -60.0))
    active = (db > threshold).astype(int)

    need = max(1, MIN_ACTIVE_MS // HOP_MS)
    sustained = np.convolve(active, np.ones(need, dtype=int), mode="valid") >= need
    hits = np.flatnonzero(sustained)
    if hits.size == 0:
        return 0, audio.size
    hop = int(sr * HOP_MS / 1000)
    frame = int(sr * FRAME_MS / 1000)
    pad = int(sr * pad_ms / 1000)
    start = hits[0] * hop
    end = (hits[-1] + need - 1) * hop + frame
    return max(0, start - pad), min(audio.size, end + pad)


def peak_normalize(audio: np.ndarray, peak_dbfs: float = TARGET_PEAK_DBFS) -> tuple[np.ndarray, float]:
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak < 1e-6:
        return audio.astype(np.float32), 0.0
    gain = (10 ** (peak_dbfs / 20)) / peak
    return (audio * gain).astype(np.float32), round(20 * np.log10(gain), 1)


def preprocess(audio: np.ndarray, sr: int = SAMPLE_RATE) -> tuple[np.ndarray, dict[str, Any]]:
    original_ms = int(round(1000 * audio.size / sr))
    filtered = highpass(audio, sr)
    start, end = speech_bounds(filtered, sr)
    trimmed = filtered[start:end]
    normalized, gain_db = peak_normalize(trimmed)
    info = {
        "highpass_hz": HIGHPASS_HZ,
        "original_duration_ms": original_ms,
        "trimmed_lead_ms": int(round(1000 * start / sr)),
        "trimmed_tail_ms": int(round(1000 * (audio.size - end) / sr)),
        "normalized_peak_dbfs": TARGET_PEAK_DBFS if gain_db else None,
        "gain_db": gain_db,
    }
    return normalized, info

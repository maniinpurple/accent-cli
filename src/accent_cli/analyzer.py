"""Acoustic + rhythm analysis.

faster-whisper gives word-level timestamps; Parselmouth (Praat) gives pitch,
intensity and formants. Everything is folded into one compact JSON payload that
an LLM coach can reason over.
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from .preprocess import preprocess
from .target import compare_to_target

SCHEMA_VERSION = "1.1"
SAMPLE_RATE = 16_000
DEFAULT_MODEL = os.getenv("ACCENT_WHISPER_MODEL", "base.en")

# Thresholds (heuristics tuned for conversational American English)
LONG_PAUSE_MS = 250          # silent gap worth flagging
LINK_GAP_MS = 120            # gap small enough to count as "linked" speech
SLOW_FN_MIN_MS = 180         # function words longer than this are suspicious
SLOW_FN_CONTENT_RATIO = 0.75  # ...or longer than 75% of the average content word
LOW_CONF = 0.60

# Pitch analysis range (covers typical male + female voices)
PITCH_FLOOR = 75.0
PITCH_CEILING = 500.0
MAX_FORMANT = 5500.0

FUNCTION_WORDS = frozenset(
    """
    a an the
    to of for in on at by from with into onto about as than like
    and but or nor so if that because
    i me my you your he him his she her it its we us our they them their
    i'm i'd i'll i've you're you'd you'll you've he's she's it's we're we'll we've
    they're they'll they've that's there's what's
    is am are was were be been being
    do does did have has had
    can could will would shall should may might must
    gonna wanna gotta
    this these those there here some any
    just not
    """.split()
)

_VOWEL_GROUPS = re.compile(r"[aeiouy]+")
_WORD_CLEAN = re.compile(r"[^a-z']")
# Intervocalic t/d inside a word (water, better, city, little, data) — not before a final silent e.
_FLAP_INSIDE = re.compile(r"[aeiou]r?(?:tt|t|dd|d)(?:[aiouy]|e(?!$)|le$)")
_ENDS_TD = re.compile(r"[aeiour](?:t|d)$")
_STARTS_VOWEL = re.compile(r"^[aeiou]")


# --------------------------------------------------------------------------- helpers

def clean_word(text: str) -> str:
    return _WORD_CLEAN.sub("", text.lower().replace("’", "'")).strip("'")


def is_function_word(text: str) -> bool:
    return clean_word(text) in FUNCTION_WORDS


def estimate_syllables(text: str) -> int:
    w = clean_word(text).replace("'", "")
    if not w:
        return 0
    groups = _VOWEL_GROUPS.findall(w)
    n = len(groups)
    if w.endswith("e") and not w.endswith(("le", "ee", "ye")) and n > 1:
        n -= 1  # silent final e
    if w.endswith("ed") and not w.endswith(("ted", "ded")) and n > 1:
        n -= 1  # -ed usually not a syllable (walked, played)
    return max(1, n)


def compute_npvi(durations: list[float]) -> float | None:
    """Normalized Pairwise Variability Index (Grabe & Low 2002).

    nPVI = 100 / (m-1) * Σ |d_k - d_{k+1}| / ((d_k + d_{k+1}) / 2)
    Higher = more alternation of long/short units (stress-timed);
    lower = more even units (syllable-timed).
    """
    d = [float(x) for x in durations if x is not None and x > 0]
    if len(d) < 2:
        return None
    terms = [abs(a - b) / ((a + b) / 2.0) for a, b in zip(d, d[1:])]
    return round(100.0 * sum(terms) / len(terms), 1)


def classify_rhythm(npvi: float | None) -> str:
    """Rough bands for word-duration nPVI (not vocalic nPVI — absolute values differ)."""
    if npvi is None:
        return "unknown"
    if npvi < 40:
        return "syllable-timed"
    if npvi < 55:
        return "mixed"
    return "stress-timed"


def _r(x: float | None, nd: int = 1) -> float | None:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return None
    return round(float(x), nd)


# --------------------------------------------------------------------------- audio / ASR

def _to_mono_16k(data: np.ndarray, sr: int) -> np.ndarray:
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != SAMPLE_RATE:
        from scipy.signal import resample_poly

        g = math.gcd(int(sr), SAMPLE_RATE)
        data = resample_poly(data, SAMPLE_RATE // g, int(sr) // g)
    return np.ascontiguousarray(data, dtype=np.float32)


def _load_wav(path: Path) -> np.ndarray:
    import warnings

    from scipy.io import wavfile

    with warnings.catch_warnings():  # macOS `say` / DAWs add harmless extra chunks
        warnings.simplefilter("ignore", wavfile.WavFileWarning)
        sr, data = wavfile.read(str(path))
    if np.issubdtype(data.dtype, np.integer):
        info = np.iinfo(data.dtype)
        data = (data.astype(np.float64) - (info.max + info.min + 1) / 2) / (info.max + 1)
    return _to_mono_16k(data.astype(np.float64), sr)


def _load_with_av(path: Path) -> np.ndarray:
    # Own PyAV decode: faster_whisper.decode_audio passes kwargs newer PyAV releases removed.
    import av

    out: list[np.ndarray] = []
    resampler = av.AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE)
    with av.open(str(path), mode="r") as container:
        stream = container.streams.audio[0]
        for frame in container.decode(stream):
            for rf in resampler.resample(frame):
                out.append(rf.to_ndarray().reshape(-1))
        for rf in resampler.resample(None):  # flush
            out.append(rf.to_ndarray().reshape(-1))
    if not out:
        return np.zeros(0, dtype=np.float32)
    return (np.concatenate(out).astype(np.float32) / 32768.0)


def load_audio(path: str | Path) -> np.ndarray:
    """Decode any audio file (wav/mp3/m4a/...) to 16 kHz mono float32."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".wav":
        try:
            return _load_wav(path)
        except ValueError:  # unusual WAV encodings — let ffmpeg handle it
            pass
    return _load_with_av(path)


@lru_cache(maxsize=2)
def get_whisper_model(name: str = DEFAULT_MODEL):
    import logging

    import ctranslate2
    from faster_whisper import WhisperModel

    ctranslate2.set_log_level(logging.ERROR)  # hide backend tips/warnings
    return WhisperModel(name, device="cpu", compute_type="int8")


def transcribe_words(audio: np.ndarray, model_name: str = DEFAULT_MODEL) -> list[dict[str, Any]]:
    """Return [{text, start, end, prob}] with times in seconds."""
    model = get_whisper_model(model_name)
    segments, _info = model.transcribe(
        audio,
        language="en",
        beam_size=5,
        word_timestamps=True,
        condition_on_previous_text=False,
        vad_filter=False,
    )
    words: list[dict[str, Any]] = []
    for seg in segments:
        for w in seg.words or []:
            text = w.word.strip()
            if not text or not clean_word(text):
                continue
            words.append(
                {"text": text, "start": float(w.start), "end": float(w.end), "prob": float(w.probability)}
            )
    return words


# --------------------------------------------------------------------------- prosody

@dataclass
class ProsodyTracks:
    pitch_t: np.ndarray
    pitch_f: np.ndarray  # 0 = unvoiced
    int_t: np.ndarray
    int_db: np.ndarray
    formant: Any  # parselmouth.Formant or None


def extract_tracks(audio: np.ndarray, sr: int = SAMPLE_RATE) -> ProsodyTracks:
    import parselmouth

    empty = np.zeros(0)
    if audio.size < sr * 0.1:
        return ProsodyTracks(empty, empty, empty, empty, None)
    snd = parselmouth.Sound(audio.astype(np.float64), sampling_frequency=sr)

    pitch = snd.to_pitch(time_step=0.01, pitch_floor=PITCH_FLOOR, pitch_ceiling=PITCH_CEILING)
    pitch_t = np.asarray(pitch.xs())
    pitch_f = np.asarray(pitch.selected_array["frequency"])

    try:
        intensity = snd.to_intensity(minimum_pitch=PITCH_FLOOR, time_step=0.01)
        int_t = np.asarray(intensity.xs())
        int_db = np.asarray(intensity.values[0])
    except parselmouth.PraatError:
        int_t, int_db = empty, empty

    try:
        formant = snd.to_formant_burg(time_step=0.01, max_number_of_formants=5, maximum_formant=MAX_FORMANT)
    except parselmouth.PraatError:
        formant = None
    return ProsodyTracks(pitch_t, pitch_f, int_t, int_db, formant)


def word_prosody(tracks: ProsodyTracks, start: float, end: float) -> dict[str, Any]:
    out: dict[str, Any] = {
        "pitch_avg_hz": None,
        "pitch_std_hz": None,
        "pitch_end_drop_hz": None,
        "voiced_pct": 0,
        "intensity_db": None,
        "f1_hz": None,
        "f2_hz": None,
    }
    if tracks.pitch_t.size:
        in_word = (tracks.pitch_t >= start) & (tracks.pitch_t <= end)
        voiced = in_word & (tracks.pitch_f > 0)
        total = int(in_word.sum())
        vf = tracks.pitch_f[voiced]
        vt = tracks.pitch_t[voiced]
        out["voiced_pct"] = int(round(100 * len(vf) / total)) if total else 0
        if len(vf) >= 3:
            out["pitch_avg_hz"] = _r(np.mean(vf))
            out["pitch_std_hz"] = _r(np.std(vf))
            k = max(1, len(vf) // 3)
            # positive = pitch falls across the word, negative = rises
            out["pitch_end_drop_hz"] = _r(np.mean(vf[:k]) - np.mean(vf[-k:]))

        if tracks.formant is not None and len(vt):
            dur = end - start
            core = vt[(vt >= start + 0.2 * dur) & (vt <= end - 0.2 * dur)]
            times = core if len(core) else vt
            f1 = [tracks.formant.get_value_at_time(1, float(t)) for t in times]
            f2 = [tracks.formant.get_value_at_time(2, float(t)) for t in times]
            f1 = [v for v in f1 if not math.isnan(v)]
            f2 = [v for v in f2 if not math.isnan(v)]
            out["f1_hz"] = int(round(np.median(f1))) if f1 else None
            out["f2_hz"] = int(round(np.median(f2))) if f2 else None

    if tracks.int_t.size:
        m = (tracks.int_t >= start) & (tracks.int_t <= end)
        if m.any():
            out["intensity_db"] = _r(np.mean(tracks.int_db[m]))
    return out


def pitch_summary(tracks: ProsodyTracks, start: float, end: float) -> dict[str, Any]:
    if not tracks.pitch_t.size:
        return {"pitch_mean_hz": None, "pitch_std_hz": None, "pitch_range_st": None}
    m = (tracks.pitch_t >= start) & (tracks.pitch_t <= end) & (tracks.pitch_f > 0)
    f = tracks.pitch_f[m]
    if len(f) < 5:
        return {"pitch_mean_hz": None, "pitch_std_hz": None, "pitch_range_st": None}
    p5, p95 = np.percentile(f, [5, 95])
    return {
        "pitch_mean_hz": _r(np.mean(f)),
        "pitch_std_hz": _r(np.std(f)),
        "pitch_range_st": _r(12 * math.log2(p95 / p5)) if p5 > 0 else None,
    }


# --------------------------------------------------------------------------- payload

def build_payload(
    audio: np.ndarray,
    raw_words: list[dict[str, Any]],
    *,
    source: str = "<audio>",
    model_name: str = DEFAULT_MODEL,
    sr: int = SAMPLE_RATE,
    target_text: str | None = None,
    preprocessing: dict[str, Any] | None = None,
) -> dict[str, Any]:
    tracks = extract_tracks(audio, sr)
    raw_words = sorted(raw_words, key=lambda w: w["start"])

    words: list[dict[str, Any]] = []
    for i, w in enumerate(raw_words):
        start_ms = int(round(w["start"] * 1000))
        end_ms = max(start_ms + 1, int(round(w["end"] * 1000)))
        if i + 1 < len(raw_words):
            pause = max(0, int(round(raw_words[i + 1]["start"] * 1000)) - end_ms)
        else:
            pause = None
        entry = {
            "i": i,
            "word": w["text"],
            "start_ms": start_ms,
            "end_ms": end_ms,
            "duration_ms": end_ms - start_ms,
            "pause_after_ms": pause,
            "syllables": estimate_syllables(w["text"]),
            "function_word": is_function_word(w["text"]),
            "conf": _r(w.get("prob"), 2),
        }
        entry.update(word_prosody(tracks, w["start"], w["end"]))
        words.append(entry)

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "meta": {
            "source": source,
            "audio_duration_ms": int(round(1000 * len(audio) / sr)),
            "sample_rate": sr,
            "asr_model": f"faster-whisper/{model_name}",
            "analyzed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "units": "ms for time, Hz for pitch/formants, dB for intensity, st = semitones",
            "preprocessing": preprocessing,
        },
        "transcript": " ".join(w["word"] for w in words),
        **compare_to_target(target_text, words),
        "summary": _summary(words, tracks),
        "flags": _flags(words),
        "words": words,
    }
    return payload


def _summary(words: list[dict[str, Any]], tracks: ProsodyTracks) -> dict[str, Any]:
    if not words:
        return {"word_count": 0, "npvi": None, "rhythm_class": "unknown"}
    durs = [w["duration_ms"] for w in words]
    fn = [w["duration_ms"] for w in words if w["function_word"]]
    ct = [w["duration_ms"] for w in words if not w["function_word"]]
    pauses = [w["pause_after_ms"] for w in words if w["pause_after_ms"] is not None]
    long_pauses = [p for p in pauses if p >= LONG_PAUSE_MS]
    span_ms = words[-1]["end_ms"] - words[0]["start_ms"]
    syllables = sum(w["syllables"] for w in words)
    fn_avg = float(np.mean(fn)) if fn else None
    ct_avg = float(np.mean(ct)) if ct else None
    npvi = compute_npvi(durs)
    confs = [w["conf"] for w in words if w["conf"] is not None]

    s = {
        "word_count": len(words),
        "syllable_count_est": syllables,
        "speech_span_ms": span_ms,
        "speech_rate_wps": _r(len(words) / (span_ms / 1000), 2) if span_ms > 0 else None,
        "articulation_rate_sps": _r(syllables / (sum(durs) / 1000), 2) if sum(durs) > 0 else None,
        "npvi": npvi,
        "rhythm_class": classify_rhythm(npvi),
        "function_word_avg_ms": _r(fn_avg, 0),
        "content_word_avg_ms": _r(ct_avg, 0),
        "function_content_ratio": _r(fn_avg / ct_avg, 2) if fn_avg and ct_avg else None,
        "pause_count": len(long_pauses),
        "pause_total_ms": int(sum(long_pauses)),
        "mean_conf": _r(float(np.mean(confs)), 2) if confs else None,
    }
    s.update(pitch_summary(tracks, words[0]["start_ms"] / 1000, words[-1]["end_ms"] / 1000))
    return s


def _flags(words: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    ct = [w["duration_ms"] for w in words if not w["function_word"]]
    ct_avg = float(np.mean(ct)) if ct else None
    slow_threshold = max(SLOW_FN_MIN_MS, SLOW_FN_CONTENT_RATIO * ct_avg) if ct_avg else SLOW_FN_MIN_MS

    slow_fn, pauses, low_conf, flaps = [], [], [], []
    for idx, w in enumerate(words):
        before_pause = w["pause_after_ms"] is None or w["pause_after_ms"] >= LONG_PAUSE_MS
        # Phrase-final lengthening is natural, so only flag function words mid-phrase.
        if w["function_word"] and not before_pause and w["duration_ms"] > slow_threshold:
            slow_fn.append({"i": w["i"], "word": w["word"], "duration_ms": w["duration_ms"]})

        if w["pause_after_ms"] is not None and w["pause_after_ms"] >= LONG_PAUSE_MS:
            pauses.append(
                {
                    "after_i": w["i"],
                    "after_word": w["word"],
                    "pause_ms": w["pause_after_ms"],
                    "at_punctuation": bool(re.search(r"[.,!?;:]$", w["word"])),
                }
            )

        if w["conf"] is not None and w["conf"] < LOW_CONF:
            low_conf.append({"i": w["i"], "word": w["word"], "conf": w["conf"]})

        cw = clean_word(w["word"])
        if _FLAP_INSIDE.search(cw):
            flaps.append({"i": w["i"], "text": w["word"], "type": "intervocalic", "duration_ms": w["duration_ms"]})
        if idx + 1 < len(words) and _ENDS_TD.search(cw):
            nxt = words[idx + 1]
            if _STARTS_VOWEL.match(clean_word(nxt["word"])) and (w["pause_after_ms"] or 0) <= LINK_GAP_MS:
                flaps.append(
                    {
                        "i": w["i"],
                        "text": f"{w['word']} {nxt['word']}",
                        "type": "cross-word",
                        "gap_ms": w["pause_after_ms"],
                    }
                )

    return {
        "slow_function_words": slow_fn,
        "long_pauses": pauses,
        "low_confidence_words": low_conf,
        "flap_candidates": flaps,
    }


# --------------------------------------------------------------------------- public API

def load_and_preprocess(path: str | Path, clean: bool = True) -> tuple[np.ndarray, dict[str, Any] | None]:
    """Decode to 16 kHz mono, then (optionally) high-pass, trim silence, normalize."""
    audio = load_audio(path)
    if not clean:
        return audio, None
    return preprocess(audio, SAMPLE_RATE)


def analyze_audio(
    audio: np.ndarray,
    *,
    source: str = "<audio>",
    model_name: str = DEFAULT_MODEL,
    target_text: str | None = None,
    clean: bool = True,
) -> dict[str, Any]:
    info = None
    if clean:
        audio, info = preprocess(audio, SAMPLE_RATE)
    raw = transcribe_words(audio, model_name)
    return build_payload(audio, raw, source=source, model_name=model_name, target_text=target_text, preprocessing=info)


def analyze_file(
    path: str | Path,
    model_name: str = DEFAULT_MODEL,
    target_text: str | None = None,
    clean: bool = True,
) -> dict[str, Any]:
    path = Path(path)
    return analyze_audio(load_audio(path), source=path.name, model_name=model_name, target_text=target_text, clean=clean)


def to_json(payload: dict[str, Any], compact: bool = True) -> str:
    if compact:
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return json.dumps(payload, ensure_ascii=False, indent=2)


def save_json(payload: dict[str, Any], export_dir: str | Path = "exports") -> Path:
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = export_dir / f"accent_{stamp}.json"
    n = 1
    while path.exists():  # two saves in the same second
        path = export_dir / f"accent_{stamp}_{n}.json"
        n += 1
    path.write_text(to_json(payload, compact=False), encoding="utf-8")
    return path

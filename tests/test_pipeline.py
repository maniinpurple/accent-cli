"""End-to-end self-test for accent-cli — no microphone needed.

Speech comes from OS TTS (macOS `say`, or espeak/espeak-ng on Linux). If no TTS
exists, a synthetic voiced signal is used and the ASR-dependent assertions are
relaxed (the schema/prosody/export checks still run).
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from accent_cli import analyzer, preprocess, prompts, target
from accent_cli.cli import app
from accent_cli.providers import PROVIDERS, ProviderError, configured_providers, get_provider
from accent_cli.recorder import save_wav

SR = analyzer.SAMPLE_RATE
SENTENCE = "I want to go to the store to get a bottle of water for my little sister."


# --------------------------------------------------------------------------- audio fixtures

def synth_voice(f0_start: float, f0_end: float, seconds: float, sr: int = SR) -> np.ndarray:
    """Harmonic-rich glide (speech-like source) with a smooth amplitude envelope."""
    n = int(seconds * sr)
    f0 = np.linspace(f0_start, f0_end, n)
    phase = 2 * np.pi * np.cumsum(f0) / sr
    sig = sum(np.sin(k * phase) / k for k in range(1, 12))
    env = np.hanning(n) ** 0.3
    return (0.4 * sig * env / np.max(np.abs(sig))).astype(np.float32)


def silence(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * SR), dtype=np.float32)


def _tts_to_wav(text: str, out: Path) -> bool:
    if shutil.which("say"):
        r = subprocess.run(["say", "-o", str(out), "--data-format=LEI16@16000", text], capture_output=True)
        return r.returncode == 0 and out.exists() and out.stat().st_size > 1000
    for exe in ("espeak-ng", "espeak"):
        if shutil.which(exe):
            r = subprocess.run([exe, "-w", str(out), text], capture_output=True)
            return r.returncode == 0 and out.exists()
    return False


@pytest.fixture(scope="session")
def speech_wav(tmp_path_factory) -> tuple[Path, bool]:
    """(path, is_real_speech)."""
    out = tmp_path_factory.mktemp("audio") / "speech.wav"
    if _tts_to_wav(SENTENCE, out):
        return out, True
    # Fallback: "syllables" of voiced glides separated by short gaps.
    parts = [silence(0.2)]
    for i in range(10):
        parts += [synth_voice(140 + 10 * (i % 3), 120, 0.12 + 0.1 * (i % 2)), silence(0.05)]
    save_wav(np.concatenate(parts), out)
    return out, False


@pytest.fixture(scope="session")
def payload(speech_wav) -> dict:
    path, _ = speech_wav
    return analyzer.analyze_file(path)


@pytest.fixture(scope="session")
def padded_wav(speech_wav, tmp_path_factory) -> Path:
    """Realistic 'mic' take: 1.0 s room noise + 40 Hz hum before, 0.8 s after the speech."""
    path, _ = speech_wav
    speech = analyzer.load_audio(path)
    rng = np.random.default_rng(0)
    lead, tail = silence(1.0), silence(0.8)
    audio = np.concatenate([lead, speech, tail])
    t = np.arange(audio.size) / SR
    audio = 0.5 * audio + 0.003 * rng.standard_normal(audio.size) + 0.02 * np.sin(2 * np.pi * 40 * t)
    out = tmp_path_factory.mktemp("audio") / "padded.wav"
    save_wav(audio.astype(np.float32), out)
    return out


# --------------------------------------------------------------------------- audio loading

def test_audio_decoders_agree(speech_wav, tmp_path):
    """scipy WAV path and PyAV (non-WAV) path give the same 16 kHz mono signal; 44.1k stereo is resampled."""
    path, _ = speech_wav
    a = analyzer._load_wav(path)
    b = analyzer._load_with_av(path)
    assert a.dtype == np.float32 and abs(len(a) - len(b)) < SR * 0.05
    n = min(len(a), len(b))
    assert np.corrcoef(a[:n], b[:n])[0, 1] > 0.99

    from scipy.io import wavfile

    stereo = np.stack([synth_voice(150, 150, 1.0, sr=44100)] * 2, axis=1)
    wavfile.write(str(tmp_path / "st.wav"), 44100, (stereo * 32767).astype(np.int16))
    mono = analyzer.load_audio(tmp_path / "st.wav")
    assert mono.ndim == 1 and abs(len(mono) - SR) < 10


# --------------------------------------------------------------------------- preprocessing

def _band_db(x: np.ndarray, hz: float) -> float:
    spec = np.abs(np.fft.rfft(x * np.hanning(x.size)))
    freqs = np.fft.rfftfreq(x.size, 1 / SR)
    return 20 * np.log10(spec[np.argmin(np.abs(freqs - hz))] + 1e-12)


def test_highpass_removes_rumble_keeps_voice():
    t = np.arange(SR * 2) / SR
    x = (0.3 * np.sin(2 * np.pi * 30 * t) + 0.3 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
    y = preprocess.highpass(x)
    assert _band_db(x, 30) - _band_db(y, 30) > 20, "30 Hz rumble should drop > 20 dB"
    assert abs(_band_db(x, 200) - _band_db(y, 200)) < 1, "200 Hz (F0 range) must be untouched"


def test_trim_removes_leading_and_trailing_silence():
    rng = np.random.default_rng(1)
    voice = synth_voice(160, 130, 0.6)
    audio = np.concatenate([silence(1.0), voice, silence(0.7)])
    audio = audio + 0.001 * rng.standard_normal(audio.size).astype(np.float32)  # -60 dBFS noise floor
    out, info = preprocess.preprocess(audio)
    assert 880 <= info["trimmed_lead_ms"] <= 990, info   # ~1000 ms minus ~50 ms pad
    assert 580 <= info["trimmed_tail_ms"] <= 690, info
    assert 0.6 <= out.size / SR <= 0.8
    assert info["original_duration_ms"] == 2300


def test_trim_ignores_short_click_and_keeps_plosive_pad():
    audio = np.concatenate([silence(0.5), synth_voice(150, 150, 0.5), silence(0.6)])
    audio[int(1.5 * SR): int(1.5 * SR) + 80] = 0.9  # 5 ms key click near the end
    start, end = preprocess.speech_bounds(preprocess.highpass(audio))
    assert start / SR == pytest.approx(0.5 - preprocess.TRIM_PAD_MS / 1000, abs=0.03)
    assert end / SR < 1.15, "a 5 ms click must not extend the speech region"


def test_peak_normalize():
    x = 0.1 * synth_voice(150, 150, 0.5)
    y, gain = preprocess.peak_normalize(x)
    assert np.max(np.abs(y)) == pytest.approx(10 ** (-3 / 20), rel=1e-3)
    assert gain > 0
    z, g0 = preprocess.peak_normalize(np.zeros(100, dtype=np.float32))
    assert g0 == 0.0 and not z.any()


# --------------------------------------------------------------------------- target text

def _words(text: str) -> list[dict]:
    return [{"i": k, "word": w, "start_ms": 100 * k} for k, w in enumerate(text.split())]


def test_target_mismatch_detection():
    r = target.compare_to_target("I want to go to the store.", _words("I want go to a store"))
    kinds = sorted(m["type"] for m in r["mismatches"])
    assert kinds == ["deletion", "substitution"]
    sub = next(m for m in r["mismatches"] if m["type"] == "substitution")
    assert (sub["expected"], sub["heard"], sub["i"], sub["at_ms"]) == ("the", "a", 4, 400)
    dele = next(m for m in r["mismatches"] if m["type"] == "deletion")
    assert dele["expected"] == "to" and dele["heard"] is None
    assert r["word_error_rate"] == pytest.approx(2 / 7, abs=1e-3)
    assert r["target_text"] == "I want to go to the store."


def test_target_insertion_and_normalization():
    r = target.compare_to_target("hello world", _words("Hello, um, world!"))
    assert r["mismatches"] == [{"type": "insertion", "expected": None, "heard": "um", "i": 1, "at_ms": 100}]
    assert r["word_error_rate"] == 0.5
    ok = target.compare_to_target("Don’t STOP—believing!", _words("don't stop believing"))
    assert ok["word_error_rate"] == 0 and ok["mismatches"] == []
    reduced = target.compare_to_target("I'm gonna go, 'cause I wanna", _words("I'm going to go because I want to"))
    assert reduced["word_error_rate"] == 0, reduced["mismatches"]
    none = target.compare_to_target(None, _words("anything"))
    assert none == {"target_text": None, "word_error_rate": None, "mismatches": []}
    everything_missing = target.compare_to_target("one two", [])
    assert everything_missing["word_error_rate"] == 1.0


# --------------------------------------------------------------------------- pure metrics

def test_npvi_known_values():
    assert analyzer.compute_npvi([100, 100, 100, 100]) == 0.0
    # |100-200| / 150 = 0.6667 -> 66.7
    assert analyzer.compute_npvi([100, 200]) == pytest.approx(66.7, abs=0.05)
    # mean of 66.67 and 66.67
    assert analyzer.compute_npvi([100, 200, 100]) == pytest.approx(66.7, abs=0.05)
    assert analyzer.compute_npvi([120]) is None
    assert analyzer.compute_npvi([0, 0, 100]) is None  # zero durations dropped


def test_rhythm_classes():
    assert analyzer.classify_rhythm(30) == "syllable-timed"
    assert analyzer.classify_rhythm(48) == "mixed"
    assert analyzer.classify_rhythm(62) == "stress-timed"
    assert analyzer.classify_rhythm(None) == "unknown"


def test_word_helpers():
    assert analyzer.is_function_word("The,")
    assert analyzer.is_function_word("I’m")
    assert not analyzer.is_function_word("water")
    assert analyzer.estimate_syllables("water") == 2
    assert analyzer.estimate_syllables("store") == 1
    assert analyzer.estimate_syllables("little") == 2
    assert analyzer.estimate_syllables("walked") == 1


# --------------------------------------------------------------------------- parselmouth prosody

def test_parselmouth_pitch_on_synthetic_words():
    """Known contours: word 0 falls 200→120 Hz, word 1 is flat 150 Hz, 100 ms gap."""
    audio = np.concatenate([silence(0.1), synth_voice(200, 120, 0.5), silence(0.1), synth_voice(150, 150, 0.3), silence(0.1)])
    raw = [
        {"text": "Hello", "start": 0.1, "end": 0.6, "prob": 0.95},
        {"text": "the", "start": 0.7, "end": 1.0, "prob": 0.40},
    ]
    p = analyzer.build_payload(audio, raw, source="synthetic")
    w0, w1 = p["words"]

    assert w0["pause_after_ms"] == 100
    assert w1["pause_after_ms"] is None
    assert w0["duration_ms"] == 500 and w1["duration_ms"] == 300
    assert 140 < w0["pitch_avg_hz"] < 180
    assert w0["pitch_end_drop_hz"] > 30, "falling contour should give a positive end drop"
    assert w0["pitch_std_hz"] > 10
    assert w1["pitch_avg_hz"] == pytest.approx(150, abs=5)
    assert abs(w1["pitch_end_drop_hz"]) < 5
    assert w1["pitch_std_hz"] < 5
    assert w0["voiced_pct"] > 70
    assert w0["intensity_db"] is not None
    assert w0["f1_hz"] is None or w0["f1_hz"] > 0
    assert w1["function_word"] and not w0["function_word"]
    assert p["summary"]["npvi"] == pytest.approx(analyzer.compute_npvi([500, 300]))
    assert p["flags"]["low_confidence_words"][0]["word"] == "the"
    assert p["summary"]["pitch_range_st"] > 3


def test_empty_audio_does_not_crash():
    p = analyzer.build_payload(silence(0.05), [], source="empty")
    assert p["words"] == [] and p["summary"]["word_count"] == 0
    assert p["transcript"] == ""


# --------------------------------------------------------------------------- full pipeline

WORD_KEYS = {
    "i", "word", "start_ms", "end_ms", "duration_ms", "pause_after_ms", "syllables", "function_word", "conf",
    "pitch_avg_hz", "pitch_std_hz", "pitch_end_drop_hz", "voiced_pct", "intensity_db", "f1_hz", "f2_hz",
}
SUMMARY_KEYS = {
    "word_count", "npvi", "rhythm_class", "function_word_avg_ms", "content_word_avg_ms",
    "function_content_ratio", "articulation_rate_sps", "speech_rate_wps", "pause_count",
    "pitch_mean_hz", "pitch_std_hz", "pitch_range_st",
}


def test_full_pipeline_schema(payload, speech_wav):
    _, real = speech_wav
    assert payload["schema_version"] == analyzer.SCHEMA_VERSION
    assert set(payload) == {
        "schema_version", "meta", "transcript", "target_text", "word_error_rate", "mismatches",
        "summary", "flags", "words",
    }
    assert payload["target_text"] is None and payload["mismatches"] == []
    assert set(payload["meta"]["preprocessing"]) >= {"highpass_hz", "trimmed_lead_ms", "trimmed_tail_ms", "gain_db"}
    assert set(payload["flags"]) == {"slow_function_words", "long_pauses", "low_confidence_words", "flap_candidates"}
    assert payload["meta"]["sample_rate"] == SR and payload["meta"]["audio_duration_ms"] > 0
    json.loads(analyzer.to_json(payload))  # serializable (no NaN / numpy types)
    assert "NaN" not in analyzer.to_json(payload)

    words = payload["words"]
    if not real:
        pytest.skip("No OS TTS available — ASR word assertions skipped (schema checked).")

    assert len(words) >= 10, payload["transcript"]
    assert SUMMARY_KEYS <= set(payload["summary"])
    assert "water" in payload["transcript"].lower()
    prev_end = 0
    for k, w in enumerate(words):
        assert set(w) == WORD_KEYS
        assert w["i"] == k
        assert w["end_ms"] > w["start_ms"] >= prev_end - 1  # monotonic (1 ms rounding slack)
        assert w["duration_ms"] == w["end_ms"] - w["start_ms"]
        if k + 1 < len(words):
            assert w["pause_after_ms"] == max(0, words[k + 1]["start_ms"] - w["end_ms"])
        prev_end = w["end_ms"]
    assert words[-1]["pause_after_ms"] is None
    assert words[-1]["end_ms"] <= payload["meta"]["audio_duration_ms"] + 50

    s = payload["summary"]
    assert s["npvi"] == analyzer.compute_npvi([w["duration_ms"] for w in words])
    assert 0 < s["npvi"] < 200
    assert s["rhythm_class"] in {"syllable-timed", "mixed", "stress-timed"}
    assert s["function_content_ratio"] > 0
    voiced = [w for w in words if w["pitch_avg_hz"] is not None]
    assert len(voiced) >= len(words) // 2, "Parselmouth should find pitch in most TTS words"
    assert all(analyzer.PITCH_FLOOR <= w["pitch_avg_hz"] <= analyzer.PITCH_CEILING for w in voiced)
    assert s["pitch_mean_hz"] is not None
    flap_texts = " ".join(c["text"].lower() for c in payload["flags"]["flap_candidates"])
    assert any(x in flap_texts for x in ("water", "bottle", "little"))


def test_pipeline_trims_padded_take(padded_wav, speech_wav):
    """Lead silence must not leak into word 0 (the 700 ms / 26 % voiced bug)."""
    _, real = speech_wav
    p = analyzer.analyze_file(padded_wav)
    prep = p["meta"]["preprocessing"]
    assert prep["trimmed_lead_ms"] >= 850, prep
    assert prep["trimmed_tail_ms"] >= 600, prep
    assert p["meta"]["audio_duration_ms"] < prep["original_duration_ms"] - 1400
    if not real:
        pytest.skip("No OS TTS available.")
    w0 = p["words"][0]
    assert w0["start_ms"] < 200 and w0["duration_ms"] < 450, w0
    assert p["words"][-1]["end_ms"] <= p["meta"]["audio_duration_ms"] + 50


def test_pipeline_target_text(speech_wav):
    path, real = speech_wav
    if not real:
        pytest.skip("No OS TTS available.")
    audio, prep = analyzer.load_and_preprocess(path)
    raw = analyzer.transcribe_words(audio)
    exact = analyzer.build_payload(audio, raw, target_text=SENTENCE, preprocessing=prep)
    assert exact["target_text"] == SENTENCE
    assert exact["word_error_rate"] <= 0.1, exact["mismatches"]

    wrong = SENTENCE.replace("water", "juice")
    p = analyzer.build_payload(audio, raw, target_text=wrong, preprocessing=prep)
    assert any(m["type"] == "substitution" and m["expected"] == "juice" and m["heard"] == "water"
               for m in p["mismatches"]), p["mismatches"]
    sub = next(m for m in p["mismatches"] if m["expected"] == "juice")
    assert p["words"][sub["i"]]["word"].lower().startswith("water")
    assert p["word_error_rate"] > exact["word_error_rate"]


# --------------------------------------------------------------------------- exports & prompts

def test_save_json(tmp_path, payload):
    p1 = analyzer.save_json(payload, tmp_path / "exports")
    p2 = analyzer.save_json(payload, tmp_path / "exports")
    assert p1 != p2 and p1.exists() and p2.exists()
    assert p1.name.startswith("accent_") and p1.suffix == ".json"
    assert json.loads(p1.read_text()) == json.loads(analyzer.to_json(payload))


def test_prompt_formatting(payload):
    js = analyzer.to_json(payload)
    combined = prompts.build_combined(js)
    assert combined.startswith(prompts.AI_COACH_PROMPT.rstrip())
    block = combined.split("```json\n", 1)[1].rsplit("\n```", 1)[0]
    assert json.loads(block) == payload
    for needle in ("nPVI", "Vietnamese", "flap", "Drills", "function_content_ratio",
                   "target_text", "mismatches", "word_error_rate", "substitution", "deletion", "Pronunciation Check"):
        assert needle in prompts.AI_COACH_PROMPT
    assert js in prompts.build_user_message(js)


# --------------------------------------------------------------------------- providers (offline)

def test_providers_interface(monkeypatch):
    assert set(PROVIDERS) == {"gemini", "claude"}
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert configured_providers() == []
    with pytest.raises(ProviderError):
        get_provider("claude")
    with pytest.raises(ProviderError):
        get_provider("nope")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("ANTHROPIC_MODEL", "my-model")
    prov = get_provider("claude")
    assert prov.model == "my-model" and configured_providers() == ["claude"]


# --------------------------------------------------------------------------- CLI

runner = CliRunner()


def test_cli_analyze_json(speech_wav):
    path, _ = speech_wav
    r = runner.invoke(app, ["analyze", str(path), "--json"])
    assert r.exit_code == 0, r.output
    data = json.loads(r.stdout)
    assert data["schema_version"] == analyzer.SCHEMA_VERSION


def test_cli_menu_copy_and_save(speech_wav, tmp_path, monkeypatch):
    path, real = speech_wav
    if not real:
        pytest.skip("Menu only appears when speech is detected.")
    copied: list[str] = []
    monkeypatch.setattr("pyperclip.copy", copied.append)
    export_dir = tmp_path / "exports"
    r = runner.invoke(app, ["analyze", str(path), "--words", "--export-dir", str(export_dir)], input="1\n2\n3\n4\nq\n")
    assert r.exit_code == 0, r.output
    assert "Transcript" in r.output and "nPVI" in r.output and "[q] Quit" in r.output
    assert len(copied) == 3
    assert copied[0].startswith(prompts.AI_COACH_PROMPT.rstrip()) and "```json" in copied[0]
    assert copied[1] == prompts.AI_COACH_PROMPT
    json.loads(copied[2])
    saved = list(export_dir.glob("accent_*.json"))
    assert len(saved) == 1 and json.loads(saved[0].read_text())["words"]


def test_cli_menu_api_without_keys(speech_wav, tmp_path, monkeypatch):
    path, real = speech_wav
    if not real:
        pytest.skip("Menu only appears when speech is detected.")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    r = runner.invoke(app, ["analyze", str(path), "--export-dir", str(tmp_path)], input="5\nq\n")
    assert r.exit_code == 0, r.output
    assert "No API key found" in r.output


def _fake_recorder(monkeypatch, wav: Path) -> list[dict]:
    """Replace the microphone with a pre-made file; returns the list of call kwargs."""
    calls: list[dict] = []

    def fake_record(out=None, console=None, target_text=None, wait_for_enter=True, **_):
        calls.append({"target_text": target_text, "wait_for_enter": wait_for_enter})
        return wav

    monkeypatch.setattr("accent_cli.recorder.record_to_wav", fake_record)
    return calls


def test_cli_record_review_and_retake(padded_wav, speech_wav, tmp_path, monkeypatch):
    _, real = speech_wav
    if not real:
        pytest.skip("No OS TTS available.")
    calls = _fake_recorder(monkeypatch, padded_wav)
    export_dir = tmp_path / "exports"
    wrong = SENTENCE.replace("water", "juice")
    # retake once, then continue, then save + quit
    r = runner.invoke(app, ["record", "--text", wrong, "--export-dir", str(export_dir)], input="r\nc\n4\nq\n")
    assert r.exit_code == 0, r.output
    assert len(calls) == 2
    assert calls[0] == {"target_text": wrong, "wait_for_enter": True}
    assert r.output.count("What the machine heard") == 2
    assert "Retake" in r.output and "WER" in r.output and "trimmed" in r.output
    saved = json.loads(next(export_dir.glob("accent_*.json")).read_text())
    assert saved["target_text"] == wrong
    assert any(m["expected"] == "juice" for m in saved["mismatches"])
    assert saved["meta"]["preprocessing"]["trimmed_lead_ms"] >= 850


def test_cli_record_interactive_target_prompt(padded_wav, speech_wav, tmp_path, monkeypatch):
    _, real = speech_wav
    if not real:
        pytest.skip("No OS TTS available.")
    calls = _fake_recorder(monkeypatch, padded_wav)
    r = runner.invoke(app, ["--help"])  # sanity: default command group still works
    assert r.exit_code == 0
    r = runner.invoke(app, ["record", "--export-dir", str(tmp_path), "--json"], input=f"{SENTENCE}\nc\n")
    assert r.exit_code == 0, r.output
    assert calls == [{"target_text": SENTENCE, "wait_for_enter": True}]  # always an explicit Enter to start
    data = json.loads(r.stdout)
    assert data["target_text"] == SENTENCE and data["word_error_rate"] <= 0.1


def test_cli_record_skip_target(padded_wav, speech_wav, tmp_path, monkeypatch):
    _, real = speech_wav
    if not real:
        pytest.skip("No OS TTS available.")
    _fake_recorder(monkeypatch, padded_wav)
    r = runner.invoke(app, ["record", "--json"], input="\nc\n")
    assert r.exit_code == 0, r.output
    data = json.loads(r.stdout)
    assert data["target_text"] is None and data["word_error_rate"] is None


def test_cli_analyze_with_text_and_raw(speech_wav):
    path, real = speech_wav
    r = runner.invoke(app, ["analyze", str(path), "--json", "--raw", "--text", SENTENCE])
    assert r.exit_code == 0, r.output
    data = json.loads(r.stdout)
    assert data["meta"]["preprocessing"] is None
    assert data["target_text"] == SENTENCE

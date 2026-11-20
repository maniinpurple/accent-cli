"""Compare what the speaker meant to say (target script) with what the ASR heard."""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from typing import Any

_NON_WORD = re.compile(r"[^a-z0-9' ]+")


# Whisper writes casual reductions out in full ("gonna" -> "going to"), so expand them on
# BOTH sides; otherwise a perfectly native "gonna" would be scored as two errors.
_REDUCTIONS = {
    "gonna": "going to", "wanna": "want to", "gotta": "got to", "hafta": "have to",
    "kinda": "kind of", "sorta": "sort of", "lotta": "lot of", "outta": "out of",
    "lemme": "let me", "gimme": "give me", "dunno": "don't know", "ya": "you",
    "cuz": "because", "'cause": "because", "cause": "because",
}


def normalize_tokens(text: str) -> list[str]:
    """Lowercase, unify apostrophes, drop punctuation, split hyphens, expand reductions."""
    text = text.lower().replace("’", "'").replace("‘", "'")
    text = re.sub(r"[-–—/]", " ", text)
    text = _NON_WORD.sub(" ", text)
    out: list[str] = []
    for t in text.split():
        t = _REDUCTIONS.get(t, t).strip("'")
        out += _REDUCTIONS.get(t, t).split()
    return [t for t in out if t]


@dataclass
class AlignOp:
    op: str                 # equal | substitution | deletion | insertion
    expected: str | None
    heard: str | None
    i: int | None           # index into words[] of the heard word (or the next heard word, for deletions)


def align(target_text: str, words: list[dict[str, Any]]) -> list[AlignOp]:
    ref = normalize_tokens(target_text)
    hyp: list[str] = []
    hyp_word_i: list[int] = []
    for w in words:
        for tok in normalize_tokens(w["word"]):
            hyp.append(tok)
            hyp_word_i.append(w["i"])

    def at(j: int) -> int | None:
        if j < len(hyp_word_i):
            return hyp_word_i[j]
        return hyp_word_i[-1] if hyp_word_i else None

    ops: list[AlignOp] = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, ref, hyp, autojunk=False).get_opcodes():
        if tag == "equal":
            ops += [AlignOp("equal", ref[i1 + k], hyp[j1 + k], hyp_word_i[j1 + k]) for k in range(i2 - i1)]
        elif tag == "delete":
            ops += [AlignOp("deletion", ref[k], None, at(j1)) for k in range(i1, i2)]
        elif tag == "insert":
            ops += [AlignOp("insertion", None, hyp[k], hyp_word_i[k]) for k in range(j1, j2)]
        else:  # replace: pair word-by-word, leftovers become deletions/insertions
            n = min(i2 - i1, j2 - j1)
            ops += [AlignOp("substitution", ref[i1 + k], hyp[j1 + k], hyp_word_i[j1 + k]) for k in range(n)]
            ops += [AlignOp("deletion", ref[k], None, at(j1 + n)) for k in range(i1 + n, i2)]
            ops += [AlignOp("insertion", None, hyp[k], hyp_word_i[k]) for k in range(j1 + n, j2)]
    return ops


def compare_to_target(target_text: str | None, words: list[dict[str, Any]]) -> dict[str, Any]:
    """JSON block: target_text, word_error_rate, mismatches[]."""
    if not target_text or not normalize_tokens(target_text):
        return {"target_text": None, "word_error_rate": None, "mismatches": []}

    start_ms = {w["i"]: w["start_ms"] for w in words}
    ops = align(target_text, words)
    mismatches = [
        {
            "type": op.op,
            "expected": op.expected,
            "heard": op.heard,
            "i": op.i,
            "at_ms": start_ms.get(op.i) if op.i is not None else None,
        }
        for op in ops
        if op.op != "equal"
    ]
    n_ref = len(normalize_tokens(target_text))
    return {
        "target_text": target_text.strip(),
        "word_error_rate": round(len(mismatches) / n_ref, 3),
        "mismatches": mismatches,
    }

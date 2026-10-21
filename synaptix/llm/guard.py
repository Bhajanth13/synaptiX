"""Guardrail for SLM rewrites: pure functions, no model needed.

A rewrite is accepted only if it adds nothing to the verified text. We check numbers, that the
key term survives, that no new proper names or childish phrases appear, that (almost) every
content word is already in the source, and that the length stays close to the source.
"""

import re
from dataclasses import dataclass, field

from synaptix.kb.store import tokens

ALLOWED = frozenset("""
the a an
you your they their it its this that someone person people
is are was be can must may will should has have do does not no
and or but if so because when where which who to of in on at by for from with as than
also only any all each every some other same there here then
means called get give go ask tell write written paper office
""".split())

BANNED = re.compile(r"\b(bad guys?|good guys?|superhero|kids?|children|imagine|friendly helpers?|like helpers?)\b|!",
                    re.I)
META_PREFIX = re.compile(r"^\s*(sure|okay|ok|certainly|here is|here's)[^:\n]*:\s*", re.I)

MAX_NOVEL = 2
MIN_SENT_SUPPORT = 0.6
MIN_COVERAGE = 0.5
LEN_MIN, LEN_MAX = 0.5, 1.5

_NUM = re.compile(r"\d+(?:[.,]\d+)*")
_WORD = re.compile(r"[A-Za-z][A-Za-z']*")


@dataclass
class Check:
    ok: bool
    reasons: list[str] = field(default_factory=list)
    novel: list[str] = field(default_factory=list)
    coverage: float = 1.0
    len_ratio: float = 1.0


def clean(text: str) -> str:
    text = META_PREFIX.sub("", text.strip())
    text = text.strip().strip("\"'“”").strip()
    return re.sub(r"\s+", " ", text)


def numbers(text: str) -> set[str]:
    return set(_NUM.findall(text))


def stem(w: str) -> str:
    w = w.lower()
    if len(w) > 4:
        for suf in ("ing", "ed", "es", "s"):
            if w.endswith(suf) and len(w) - len(suf) >= 3:
                return w[: -len(suf)]
    return w


def content_stems(text: str) -> set[str]:
    return {stem(t) for t in tokens(text) if t not in ALLOWED and not t.isdigit()}


def _has_phrase(text: str, phrase: str) -> bool:
    return re.search(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", text, re.I) is not None


def _new_entities(source: str, rewrite: str) -> list[str]:
    out = []
    for m in _WORD.finditer(rewrite):
        w = m.group()
        if len(w) < 2:
            continue
        before = rewrite[: m.start()].rstrip(" \t(\"'“")
        starts_sentence = before == "" or before[-1] in ".?!;:\n"
        if w.isupper():
            if not re.search(r"(?<!\w)" + re.escape(w) + r"(?!\w)", source):
                out.append(w)
        elif w[0].isupper() and not starts_sentence:
            if not _has_phrase(source, w):
                out.append(w)
    return out


def check_rewrite(source: str, rewrite: str, term: str | None = None, aliases=()) -> Check:
    reasons: list[str] = []
    if not rewrite or not rewrite.strip():
        return Check(False, ["empty"], [], 0.0, 0.0)
    if numbers(rewrite) != numbers(source):
        reasons.append("numbers")
    known = source + " " + (term or "")
    if term and not any(_has_phrase(rewrite, x) for x in (term, *aliases)):
        reasons.append("term_missing")
    for tok in _new_entities(known, rewrite):
        reasons.append(f"new_entity:{tok}")
    # A banned word is fine when the verified text itself uses it (e.g. "children" in the legal-aid card).
    if any(not re.search(re.escape(m.group(0)), known, re.I) for m in BANNED.finditer(rewrite)):
        reasons.append("banned_phrase")

    src = content_stems(known)
    rw = content_stems(rewrite)
    novel = sorted(rw - src)
    if len(novel) > MAX_NOVEL:
        reasons.append("novel_words")
    for sent in re.split(r"[.?!;]", rewrite):
        stems = content_stems(sent)
        if stems and len(stems & src) / len(stems) < MIN_SENT_SUPPORT:
            reasons.append("unsupported_sentence")
            break

    src_words, rw_words = len(source.split()), len(rewrite.split())
    len_ratio = rw_words / src_words if src_words else 1.0
    if len_ratio < LEN_MIN:
        reasons.append("too_short")
    elif len_ratio > LEN_MAX:
        reasons.append("too_long")
    src_only = content_stems(source)
    coverage = len(src_only & rw) / len(src_only) if src_only else 1.0
    if coverage < MIN_COVERAGE:
        reasons.append("dropped_facts")
    return Check(not reasons, reasons, novel, coverage, len_ratio)

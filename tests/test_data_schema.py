import re
from pathlib import Path

import pytest

from synaptix.core.location import normalize
from synaptix.kb.store import DATA_DIR, KnowledgeBase
from synaptix.llm.evaluate import evaluate
from synaptix.llm.rewrites import DEFAULT_PATH, RewriteStore
from synaptix.llm.slm import SLM

KB = KnowledgeBase()
ST = KB.states["karnataka"]
ALL_CARDS = KB.legal_cards + [c for s in KB.states.values() for c in s["legal_cards"]]


def test_entries_have_source_and_review():
    for e in KB.glossary + ALL_CARDS:
        assert e.get("source") and e.get("review") in {"pending", "verified"}, e.get("term") or e.get("id")
    ids = [c["id"] for c in ALL_CARDS]
    assert len(ids) == len(set(ids))


def test_office_references():
    for c in ALL_CARDS:
        for k in c.get("escalate_offices", []):
            assert k in ST["office_types"], (c["id"], k)
    topics = {t["topic"] for t in KB.jurisdiction["topics"]}
    assert set(ST["topic_offices"]) <= topics
    for keys in ST["topic_offices"].values():
        assert set(keys) <= set(ST["office_types"])


def test_districts():
    ds = ST["districts"]
    assert len(ds) == 31 and len({d["id"] for d in ds}) == 31
    assert set(ST["office_types"]) == {"dlsa", "labour", "dc", "police", "zp"}
    seen = {}
    for d in ds:
        assert {"id", "name", "name_kn", "aliases", "website", "verify", "offices"} <= set(d)
        for n in [d["id"], d["name"], d["name_kn"], *d["aliases"]]:
            if n:
                assert seen.setdefault(normalize(n), d["id"]) == d["id"], n
        if not d["website"]:
            assert d["verify"] is True
        else:
            assert d["website"].startswith("https://") and (".nic.in" in d["website"] or ".gov.in" in d["website"])
        for o in list(d["offices"].values()) + list(ST["office_types"].values()):
            for f in ("phone", "address"):
                assert not o.get(f) or d["verify"]


def test_no_invented_numbers():
    allowed = {"15100"}  # national legal aid helpline, the only 5+ digit value in state data
    for f in (DATA_DIR / "states").rglob("*.yaml"):
        text = re.sub(r"https?://\S+", "", f.read_text(encoding="utf-8"))
        for m in re.findall(r"\d{5,}", text):
            assert m in allowed, (f.name, m)
        assert not re.search(r"\b\d{10}\b|\b0\d{2,4}-\d{6,8}\b", text), f.name


def test_prerendered_rewrites_pass_guard():
    rows = evaluate(KB, SLM("/nonexistent", store=RewriteStore(DEFAULT_PATH)), prerendered_only=True)
    assert all(r.ok for r in rows if "not_rendered" not in r.reasons)

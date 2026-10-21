"""Score SLM rewrites of the verified KB text with the guard, and prerender passing ones.

    python -m synaptix.llm.evaluate [--prerendered]
"""

import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

from synaptix.llm import guard, rewrites
from synaptix.llm.slm import PROMPT_VERSION, SLM


@dataclass
class Row:
    ref: str
    term: str | None
    ok: bool
    reasons: list[str] = field(default_factory=list)
    novel: list[str] = field(default_factory=list)
    coverage: float = 0.0
    len_ratio: float = 0.0
    seconds: float = 0.0
    text: str = ""


def items(kb):
    """(ref, term, aliases, source_text, entry_source) for every glossary `plain` and card `summary`."""
    out = [(f"glossary:{g['term']}", g["term"], g.get("aliases", []), g["plain"], g["source"])
           for g in kb.glossary]
    cards = kb.legal_cards + [c for s in kb.states.values() for c in s["legal_cards"]]
    out += [(f"legal:{c['id']}", None, [], c["summary"], c["source"]) for c in cards]
    return out


def evaluate(kb, slm: SLM, prerendered_only: bool = False) -> list[Row]:
    rows = []
    for ref, term, aliases, src, _ in items(kb):
        t0 = time.time()
        if prerendered_only:
            text = slm.store.get(src, term)
            if text is None:
                rows.append(Row(ref, term, False, ["not_rendered"]))
                continue
            how = "prerendered"
        else:
            text, _, how = slm.rewrite(src, term, aliases)
        c = guard.check_rewrite(src, text, term, aliases)
        reasons = list(c.reasons)
        if how == "passthrough":  # no model and nothing prerendered: nothing was actually scored
            reasons.append("not_rendered")
        rows.append(Row(ref, term, c.ok and how != "passthrough", reasons, c.novel, c.coverage, c.len_ratio,
                        time.time() - t0, text))
    return rows


def summary(rows: list[Row]) -> str:
    n = len(rows) or 1
    fails = Counter(r.split(":")[0] for row in rows for r in row.reasons)
    top = ", ".join(f"{k} x{v}" for k, v in fails.most_common(5)) or "none"
    return (f"{sum(r.ok for r in rows)}/{len(rows)} pass ({100 * sum(r.ok for r in rows) / n:.0f}%); "
            f"mean coverage {sum(r.coverage for r in rows) / n:.2f}; "
            f"mean seconds {sum(r.seconds for r in rows) / n:.2f}; top failures: {top}")


def prerender(kb, slm: SLM, path, force: bool = False) -> tuple[int, list[tuple[str, list[str]]]]:
    """Run the model over every entry, keep guard-passing rewrites, write them as YAML."""
    path = Path(path)
    kept = {} if force else {e["key"]: e for e in rewrites.RewriteStore(path).all()}
    written, rejected = 0, []
    live_store, slm.store = slm.store, rewrites.RewriteStore("/nonexistent/rewrites.yaml")  # always ask the model
    for ref, term, aliases, src, entry_source in items(kb):
        k = rewrites.key(src, term)
        if k in kept:
            continue
        rewrites.CACHE.pop(k, None)
        text, check, how = slm.rewrite(src, term, aliases)
        if how == "model":
            kept[k] = {"key": k, "ref": ref, "term": term, "text": text, "source": entry_source,
                       "review": "pending"}
            written += 1
        else:
            rejected.append((ref, check.reasons if check else [how]))
    slm.store = live_store
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {"model": slm._path.name, "prompt_version": PROMPT_VERSION,
           "entries": sorted(kept.values(), key=lambda e: e["ref"])}
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(doc, f, allow_unicode=True, sort_keys=False, width=100)
    return written, rejected


def main(argv=None):
    import argparse
    import json
    from synaptix.kb.store import KnowledgeBase
    p = argparse.ArgumentParser()
    p.add_argument("--prerendered", action="store_true")
    p.add_argument("--json", action="store_true")
    a = p.parse_args(argv)
    rows = evaluate(KnowledgeBase(), SLM(), prerendered_only=a.prerendered)
    if a.json:
        print(json.dumps([asdict(r) for r in rows], ensure_ascii=False, indent=1))
    else:
        for r in rows:
            print(f"{'ok  ' if r.ok else 'FAIL'} {r.ref:40} {','.join(r.reasons)}")
        print(summary(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

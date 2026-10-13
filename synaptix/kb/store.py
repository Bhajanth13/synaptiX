"""Verified knowledge base: YAML/JSON files in data/, plus a tiny lexical retriever.

The retriever is deliberately dependency-free (token overlap with IDF weighting) so it runs
on any CPU. Swap in a multilingual embedding model behind the same `search` signature later.
"""

import json
import math
import re
from functools import cached_property
from pathlib import Path

import yaml

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
_TOKEN = re.compile(r"[\w']+", re.UNICODE)
_STOP = {"the", "a", "an", "is", "of", "to", "in", "my", "i", "what", "does", "do", "this", "and",
         "for", "on", "it", "me", "who", "how", "can", "about", "mean", "means",
         # too generic to pick a legal card on their own
         "legal", "law", "laws", "rules", "rule", "rights", "right", "help"}


def tokens(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


class KnowledgeBase:
    def __init__(self, data_dir: Path = DATA_DIR):
        self.dir = Path(data_dir)

    def _yaml(self, name):
        with open(self.dir / name, encoding="utf-8") as f:
            return yaml.safe_load(f)

    @cached_property
    def glossary(self) -> list[dict]:
        return self._yaml("glossary.yaml")

    @cached_property
    def legal_cards(self) -> list[dict]:
        return self._yaml("legal_cards.yaml")

    @cached_property
    def jurisdiction(self) -> dict:
        return self._yaml("jurisdiction.yaml")

    @cached_property
    def events(self) -> list[dict]:
        with open(self.dir / "events.json", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def search(query: str, docs: list[dict], fields: list[str], k: int = 3) -> list[tuple[float, dict]]:
        """Rank docs by IDF-weighted token overlap with the query."""
        doc_toks = [set(tokens(" ".join(str(d.get(f, "")) for f in fields))) for d in docs]
        n = len(docs)
        df: dict[str, int] = {}
        for toks in doc_toks:
            for t in toks:
                df[t] = df.get(t, 0) + 1
        q = set(tokens(query))
        scored = []
        for d, toks in zip(docs, doc_toks):
            s = sum(math.log(1 + n / df[t]) for t in q & toks)
            if s > 0:
                scored.append((s, d))
        scored.sort(key=lambda x: -x[0])
        return scored[:k]

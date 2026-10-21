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
# \w misses Indic combining signs (matras, virama), which would split words apart; the extra
# range covers the Devanagari..Sinhala blocks plus ZWNJ/ZWJ used inside Kannada/Hindi words.
_TOKEN = re.compile(r"[\w\u0900-\u0DFF\u200c\u200d']+", re.UNICODE)
_STOP = {"the", "a", "an", "is", "of", "to", "in", "my", "i", "what", "does", "do", "this", "and",
         "for", "on", "it", "me", "who", "how", "can", "about", "mean", "means",
         # too generic to pick a legal card on their own
         "legal", "law", "laws", "rules", "rule", "rights", "right", "help",
         # question words in Hindi and Kannada
         "क्या", "है", "होता", "कौन", "किससे", "मेरा", "मुझे",
         "ಏನು", "ಎಂದರೇನು", "ಎಂದರೆ", "ಯಾರು", "ಯಾರಿಗೆ", "ನನ್ನ", "ಹೇಗೆ"}


def tokens(text: str) -> list[str]:
    # Strip edge apostrophes (quotes, not part of the word); keep inner ones like "people's".
    toks = (t.strip("'") for t in _TOKEN.findall(text.lower()))
    return [t for t in toks if t and t not in _STOP]


def _text(v) -> str:
    return " ".join(map(str, v)) if isinstance(v, (list, tuple)) else str(v)


class KnowledgeBase:
    def __init__(self, data_dir: Path = DATA_DIR):
        self.dir = Path(data_dir)

    def _yaml(self, name):
        with open(self.dir / name, encoding="utf-8") as f:
            return yaml.safe_load(f)

    @cached_property
    def states(self) -> dict[str, dict]:
        """{state_id: {meta, legal_cards, glossary, districts, office_types, topic_offices}}."""
        out: dict[str, dict] = {}
        root = self.dir / "states"
        for d in sorted(p for p in root.glob("*") if p.is_dir()) if root.exists() else []:
            def load(name, default):
                f = d / name
                if not f.exists():
                    return default
                with open(f, encoding="utf-8") as fh:
                    return yaml.safe_load(fh) or default
            sid = d.name
            dist = load("districts.yaml", {})
            out[sid] = {
                "meta": {"id": sid, **load("state.yaml", {})},
                "legal_cards": [{**c, "state": sid} for c in load("legal_cards.yaml", [])],
                "glossary": [{**g, "state": sid} for g in load("glossary.yaml", [])],
                "districts": dist.get("districts", []),
                "office_types": dist.get("office_types", {}),
                "topic_offices": load("jurisdiction.yaml", {}).get("topic_offices", {}),
            }
        return out

    @cached_property
    def glossary(self) -> list[dict]:
        return self._yaml("glossary.yaml") + [g for s in self.states.values() for g in s["glossary"]]

    @cached_property
    def legal_cards(self) -> list[dict]:
        """National cards only; use legal_cards_for(state) to include a state's cards."""
        return self._yaml("legal_cards.yaml")

    def legal_cards_for(self, state: str | None) -> list[dict]:
        return self.legal_cards + (self.states[state]["legal_cards"] if state in self.states else [])

    def find_district(self, name: str):
        from synaptix.core.location import normalize
        n = normalize(name)
        for sid, s in self.states.items():
            for d in s["districts"]:
                if n in {normalize(x) for x in [d["id"], d.get("name", ""), d.get("name_kn", ""),
                                                *(d.get("aliases") or [])] if x}:
                    return sid, d
        return None

    def find_state(self, name: str) -> str | None:
        from synaptix.core.location import normalize
        n = normalize(name)
        for sid, s in self.states.items():
            m = s["meta"]
            if n in {normalize(x) for x in [sid, m.get("name", ""), m.get("name_kn", ""),
                                            *(m.get("aliases") or [])] if x}:
                return sid
        return None

    def _district(self, state: str, district_id: str) -> dict:
        return next(d for d in self.states[state]["districts"] if d["id"] == district_id)

    def office(self, state: str, district_id: str, key: str) -> dict:
        """Office template merged with the district's override; name gets the district appended."""
        s, d = self.states[state], self._district(state, district_id)
        override = d.get("offices", {}).get(key) or {}
        merged = {**s["office_types"][key], **override}
        if "name" not in override:
            merged["name"] = f"{merged['name']}, {d['name']}"
        return merged

    _HELPLINE = {"dlsa": "legal aid helpline", "police": "emergency helpline"}

    def offices_text(self, state: str, district_id: str, keys: list[str]) -> str:
        parts = []
        for k in keys:
            o = self.office(state, district_id, k)
            line = o["name"]
            if o.get("helpline"):
                line += f" ({self._HELPLINE.get(k, 'helpline')} {o['helpline']})"
            parts.append(line)
        text = "; ".join(parts)
        if not self._district(state, district_id).get("website"):
            text += " (check contact details at the district office)"
        return text

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
        doc_toks = [set(tokens(" ".join(_text(d.get(f, "")) for f in fields))) for d in docs]
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

"""Prerendered plain-English rewrites of verified entries, plus an in-memory cache.

Stored rewrites are keyed by a hash of (term, source text), so editing the verified text
invalidates its rewrite automatically. No user text is ever stored.
"""

import hashlib
import os
from pathlib import Path

import yaml

from synaptix.kb.store import DATA_DIR

DEFAULT_PATH = DATA_DIR / "rewrites" / "en.yaml"

CACHE: dict[str, str] = {}  # session cache, keyed by key(); never written to disk


def key(source: str, term: str | None = None) -> str:
    return hashlib.sha256(f"{term or ''}|{source.strip()}".encode("utf-8")).hexdigest()[:16]


class RewriteStore:
    def __init__(self, path=None):
        self.path = Path(path or os.environ.get("SYNAPTIX_REWRITES") or DEFAULT_PATH)
        self._by_key: dict[str, dict] = {}
        if self.path.exists():
            with open(self.path, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            self._by_key = {e["key"]: e for e in data.get("entries") or []}

    def get(self, source: str, term: str | None = None) -> str | None:
        e = self._by_key.get(key(source, term))
        return e["text"] if e else None

    def all(self) -> list[dict]:
        return list(self._by_key.values())

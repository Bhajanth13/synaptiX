"""Translation. English is the internal pivot language.

Order of preference:
1. Pre-verified translations stored on the knowledge-base entry (`text_hi`, `text_kn`, ...).
2. IndicTrans2 (offline, 22 languages) if installed: plug it into `IndicTrans2Backend`.
3. Otherwise the English text is returned with `translated=False`, so the caller can say so.
"""


class Translator:
    def __init__(self, backend=None):
        self.backend = backend

    def to_english(self, text: str, lang: str) -> str:
        if lang == "en" or self.backend is None:
            return text
        return self.backend.translate(text, src=lang, tgt="en")

    def from_entry(self, entry: dict, field: str, lang: str) -> tuple[str, bool]:
        """Pick the verified translation of entry[field] for lang if present."""
        if lang != "en" and entry.get(f"{field}_{lang}"):
            return entry[f"{field}_{lang}"], True
        if lang == "mwr" and entry.get(f"{field}_hi"):
            return entry[f"{field}_hi"], True  # Marwari speakers broadly understand Hindi
        return entry[field], lang == "en"

    def from_english(self, text: str, lang: str) -> tuple[str, bool]:
        if lang == "en":
            return text, True
        if self.backend is None:
            return text, False
        return self.backend.translate(text, src="en", tgt=lang), True


class IndicTrans2Backend:
    """Placeholder for AI4Bharat IndicTrans2 (distilled 200M). See docs/ARCHITECTURE.md."""

    def translate(self, text: str, src: str, tgt: str) -> str:
        raise NotImplementedError("Install IndicTrans2 and wire it up here.")

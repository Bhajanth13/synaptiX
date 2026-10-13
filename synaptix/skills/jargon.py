"""Political and legal jargon -> plain words. Explains terms and processes only, never opinions."""

from synaptix.core.types import Answer
from synaptix.skills.base import Skill


class JargonSkill(Skill):
    name = "jargon"

    def answer(self, q, english):
        hits = self.kb.search(english, self.kb.glossary, ["term", "aliases"], k=1)
        if not hits:
            hits = self.kb.search(english, self.kb.glossary, ["term", "aliases", "plain"], k=1)
        if not hits:
            return self.no_match(q)
        entry = hits[0][1]
        text, ok = self.tr.from_entry(entry, "plain", q.lang)
        if q.lang == "en":
            text = self.slm.simplify(text, english)
        body = f"{entry['term']}: {text}"
        if entry.get("example"):
            body += f"\nExample: {entry['example']}"
        return Answer(body, lang=q.lang if ok else "en", skill=self.name, sources=[entry["source"]],
                      note=None if ok else f"(Shown in English: no verified {q.lang} text yet.)")

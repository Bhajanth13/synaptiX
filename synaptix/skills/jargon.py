"""Political and legal jargon -> plain words. Explains terms and processes only, never opinions."""

from synaptix.core.types import Answer
from synaptix.skills.base import Skill


class JargonSkill(Skill):
    name = "jargon"

    def answer(self, q, english):
        hits = self.kb.search(english, self.kb.glossary, ["term", "aliases"], k=1)
        closest = False  # matched only through the definition text, not the term itself
        if not hits:
            hits = self.kb.search(english, self.kb.glossary, ["term", "aliases", "plain"], k=1)
            closest = bool(hits)
        if not hits:
            return self.no_match(q)
        entry = hits[0][1]
        text, ok = self.tr.from_entry(entry, "plain", q.lang)
        if q.lang == "en":
            text = self.slm.simplify(text, english, term=entry["term"], aliases=entry.get("aliases", []))
        body = f"{entry['term']}: {text}"
        if closest and q.lang == "en":
            body = "Closest verified entry: " + body
        if entry.get("example"):
            body += f"\nExample: {entry['example']}"
        note = None if ok else f"(Shown in English: no verified {q.lang} text yet.)"
        if closest and q.lang != "en":  # English bodies carry the prefix; other languages get a note
            note = "(Closest verified entry; it may not match your question.)" + (f"\n{note}" if note else "")
        return Answer(body, lang=q.lang if ok else "en", skill=self.name, sources=[entry["source"]], note=note)

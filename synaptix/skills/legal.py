"""Legal information from expert-verified cards. Information, never advice."""

from synaptix.core.types import Answer
from synaptix.skills.base import Skill

MIN_SCORE = 1.5
MIN_TITLE_SCORE = 1.0


class LegalSkill(Skill):
    name = "legal"

    def answer(self, q, english):
        # Title/keyword matches are strong evidence; matches only in the summary text need a higher score.
        hits = self.kb.search(english, self.kb.legal_cards, ["title", "keywords"], k=1)
        if not hits or hits[0][0] < MIN_TITLE_SCORE:
            hits = self.kb.search(english, self.kb.legal_cards, ["title", "keywords", "summary"], k=1)
            if not hits or hits[0][0] < MIN_SCORE:
                return self.no_match(q)
        card = hits[0][1]
        text, ok = self.tr.from_entry(card, "summary", q.lang)
        if q.lang == "en":
            text = self.slm.simplify(text, english)
        steps = card.get("what_to_do") or []
        if steps and q.lang == "en":
            text += "\nWhat you can do:\n" + "\n".join(f"- {s}" for s in steps)
        return Answer(text, lang=q.lang if ok else "en", skill=self.name, sources=[card["source"]],
                      escalate_to=card.get("escalate_to"),
                      note=None if ok else f"(Shown in English: no verified {q.lang} text yet.)")

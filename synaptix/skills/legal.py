"""Legal information from expert-verified cards. Information, never advice."""

from synaptix.core.location import Location
from synaptix.core.types import Answer
from synaptix.skills.base import Skill

MIN_SCORE = 1.5
MIN_TITLE_SCORE = 1.0
PREFER_STATE = 0.75  # a state card wins if it scores at least this fraction of the top national card
HEADINGS = {"en": "What you can do:", "hi": "आप क्या कर सकते हैं:", "kn": "ನೀವು ಏನು ಮಾಡಬಹುದು:"}


class LegalSkill(Skill):
    name = "legal"

    def _state(self, q) -> str | None:
        loc = q.location or Location()
        if loc.state:
            return loc.state
        # No saved/explicit location: a query that names a state still gets that state's cards.
        for sid, s in self.kb.states.items():
            m = s["meta"]
            if any(n and n.lower() in q.text.lower() for n in (m.get("name"), m.get("name_kn"))):
                return sid
        return None

    def _best(self, english, cards):
        # Title/keyword matches are strong evidence; matches only in the summary text need a higher score.
        hits = self.kb.search(english, cards, ["title", "keywords"], k=5)
        if not hits or hits[0][0] < MIN_TITLE_SCORE:
            hits = self.kb.search(english, cards, ["title", "keywords", "summary"], k=5)
            if not hits or hits[0][0] < MIN_SCORE:
                return None
        top = hits[0]
        if not top[1].get("state"):
            for score, card in hits[1:]:
                if card.get("state") and score >= PREFER_STATE * top[0]:
                    return card
        return top[1]

    def answer(self, q, english):
        state = self._state(q)
        card = self._best(english, self.kb.legal_cards_for(state))
        if card is None:
            return self.no_match(q)
        text, ok = self.tr.from_entry(card, "summary", q.lang)
        if q.lang == "en":
            text = self.slm.simplify(text, english, term=None)
        lang = q.lang if ok else "en"
        steps = card.get("what_to_do") if q.lang == "en" else (
            (card.get(f"what_to_do_{q.lang}") or (card.get("what_to_do_hi") if q.lang == "mwr" else None))
            if ok else None)
        if steps:
            heading = HEADINGS.get(q.lang if q.lang in HEADINGS else "hi" if q.lang == "mwr" else "en")
            text += f"\n{heading}\n" + "\n".join(f"- {s}" for s in steps)
        loc = q.location or Location()
        escalate = card.get("escalate_to")
        if loc.district and card.get("escalate_offices"):
            escalate = self.kb.offices_text(loc.state, loc.district, card["escalate_offices"])
        source = card["source"] + (f" ({card['source_url']})" if card.get("source_url") else "")
        return Answer(text, lang=lang, skill=self.name, sources=[source], escalate_to=escalate,
                      note=None if ok else f"(Shown in English: no verified {q.lang} text yet.)")

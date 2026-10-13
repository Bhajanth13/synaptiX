from synaptix.core import language, router
from synaptix.core.translate import Translator
from synaptix.core.types import Answer, Query
from synaptix.kb.store import KnowledgeBase
from synaptix.llm.slm import SLM
from synaptix.skills.events import EventsSkill
from synaptix.skills.jargon import JargonSkill
from synaptix.skills.jurisdiction import JurisdictionSkill
from synaptix.skills.legal import LegalSkill


class Pipeline:
    """input text (any supported language) -> route -> skill (verified KB) -> answer in that language."""

    def __init__(self, kb: KnowledgeBase | None = None, slm: SLM | None = None,
                 translator: Translator | None = None):
        self.kb = kb or KnowledgeBase()
        self.slm = slm or SLM()
        self.tr = translator or Translator()
        args = (self.kb, self.slm, self.tr)
        self.skills = {s.name: s for s in (JargonSkill(*args), LegalSkill(*args),
                                           JurisdictionSkill(*args), EventsSkill(*args))}

    def ask(self, text: str, lang: str | None = None, skill: str | None = None) -> Answer:
        q = Query(text=text, lang=lang or language.detect(text))
        english = self.tr.to_english(q.text, q.lang)
        name = skill or router.route(english)
        return self.skills[name].answer(q, english)

    def ask_signs(self, glosses: list[str], lang: str = "en") -> Answer:
        """Entry point for the sign-language input: glosses -> sentence -> normal pipeline."""
        sentence = self.slm.glosses_to_sentence(glosses)
        ans = self.ask(sentence, lang="en")
        if lang != "en":
            ans.text, ok = self.tr.from_english(ans.text, lang)
            ans.lang = lang if ok else "en"
        return ans

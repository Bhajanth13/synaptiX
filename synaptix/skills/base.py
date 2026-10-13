from synaptix.core.translate import Translator
from synaptix.core.types import Answer, Query
from synaptix.kb.store import KnowledgeBase
from synaptix.llm.slm import SLM


class Skill:
    name = "base"

    def __init__(self, kb: KnowledgeBase, slm: SLM, translator: Translator):
        self.kb, self.slm, self.tr = kb, slm, translator

    def answer(self, q: Query, english: str) -> Answer:
        raise NotImplementedError

    def no_match(self, q: Query) -> Answer:
        return Answer(
            text="I don't have verified information on this yet. Please ask your local para-legal "
                 "volunteer or the District Legal Services Authority (free legal aid).",
            lang="en", skill=self.name, verified=True, escalate_to="DLSA / para-legal volunteer",
        )

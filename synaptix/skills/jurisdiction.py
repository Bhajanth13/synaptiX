"""Who holds the power: Gram Panchayat / Sarpanch vs MLA (state) vs MP (Union).

Deterministic lookup over data/jurisdiction.yaml (11th Schedule + 7th Schedule of the
Constitution). No model is involved, so it cannot hallucinate.
"""

from synaptix.core.types import Answer
from synaptix.skills.base import Skill


class JurisdictionSkill(Skill):
    name = "jurisdiction"

    def answer(self, q, english):
        data = self.kb.jurisdiction
        topics = data["topics"]
        hits = self.kb.search(english, topics, ["topic", "keywords"], k=1)
        if not hits:
            return self._overview(q)
        t = hits[0][1]
        lines = [f"For {t['topic']}:"]
        for i, step in enumerate(t["who"], 1):
            lines.append(f"{i}. {data['levels'][step['level']]['label']}: {step['role']}")
        if t.get("not_them"):
            lines.append("Not their job: " + t["not_them"])
        return Answer("\n".join(lines), lang="en", skill=self.name, sources=t["sources"],
                      note=None if q.lang == "en" else "(Shown in English: translation pending.)")

    def _overview(self, q):
        levels = self.kb.jurisdiction["levels"]
        text = "\n".join(f"- {v['label']}: {v['summary']}" for v in levels.values())
        return Answer(text, lang="en", skill=self.name, sources=["Constitution of India, 7th & 11th Schedules"])

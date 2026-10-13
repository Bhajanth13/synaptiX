"""Upcoming civic events from data/events.json (filled by synced update bundles or a local volunteer)."""

from datetime import date

from synaptix.core.types import Answer
from synaptix.skills.base import Skill


class EventsSkill(Skill):
    name = "events"

    def answer(self, q, english, today: date | None = None):
        today = today or date.today()
        upcoming = sorted((e for e in self.kb.events if date.fromisoformat(e["date"]) >= today),
                          key=lambda e: e["date"])
        if not upcoming:
            return Answer("No upcoming events are loaded. Events arrive with the next update bundle.",
                          lang="en", skill=self.name)
        hits = self.kb.search(english, upcoming, ["title", "type", "place"], k=3)
        chosen = [d for _, d in hits] or upcoming[:3]
        text = "\n".join(f"- {e['date']}: {e['title']} at {e['place']}" for e in chosen)
        return Answer(text, lang="en", skill=self.name, sources=sorted({e["source"] for e in chosen}))

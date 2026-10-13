from dataclasses import dataclass, field


@dataclass
class Query:
    text: str
    lang: str = "en"
    source: str = "text"  # text | voice | sign | ocr


@dataclass
class Answer:
    text: str
    lang: str
    skill: str
    sources: list[str] = field(default_factory=list)
    verified: bool = True  # False when the SLM paraphrased something with no verified source
    escalate_to: str | None = None  # named human / office for high-stakes cases
    note: str | None = None

    def render(self) -> str:
        out = [self.text]
        if self.sources:
            out.append("Source: " + "; ".join(self.sources))
        if not self.verified:
            out.append("(Unverified: please check with a para-legal volunteer.)")
        if self.escalate_to:
            out.append("Talk to: " + self.escalate_to)
        if self.note:
            out.append(self.note)
        return "\n".join(out)

"""Small language model wrapper (llama.cpp, CPU-only, 4-bit GGUF).

The SLM never supplies facts. It only rewrites verified text into plain language, or turns
sign-language glosses into a sentence. If no model is installed, every call passes the
input through unchanged, so the rest of the pipeline still works.
"""

import os
import re
from pathlib import Path

DEFAULT_MODEL = Path(__file__).resolve().parents[2] / "models" / "slm.gguf"

SYSTEM = (
    "You rewrite civic and legal information for rural Indian citizens. "
    "Use short sentences and everyday words a person with primary-school education understands. "
    "Only use facts from the given text; never add new facts, numbers or laws. "
    "Never give an opinion about any political party or politician. "
    "This is legal information, not legal advice."
)


class SLM:
    def __init__(self, model_path: str | None = None, n_ctx: int = 2048):
        path = Path(model_path or os.environ.get("SYNAPTIX_SLM", DEFAULT_MODEL))
        self.llm = None
        if path.exists():
            try:
                from llama_cpp import Llama
                self.llm = Llama(model_path=str(path), n_ctx=n_ctx, n_threads=os.cpu_count(), verbose=False)
            except ImportError:
                pass

    @property
    def available(self) -> bool:
        return self.llm is not None

    def _chat(self, user: str, max_tokens: int = 200) -> str:
        out = self.llm.create_chat_completion(
            messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
            max_tokens=max_tokens, temperature=0.2,
        )
        return out["choices"][0]["message"]["content"].strip()

    def simplify(self, verified_text: str, question: str) -> str:
        if not self.available:
            return verified_text
        out = self._chat(f"Question: {question}\n\nVerified text:\n{verified_text}\n\n"
                         "Answer the question in 2-4 simple sentences using only the verified text. "
                         "Keep every number exactly as written.")
        # Guardrail: a rewrite that drops, changes or invents a number is rejected.
        return out if _numbers(out) == _numbers(verified_text) else verified_text

    def glosses_to_sentence(self, glosses: list[str]) -> str:
        """ISL gloss sequence (e.g. LAND NOTICE MEANING WHAT) -> English question."""
        if not self.available:
            return " ".join(g.lower() for g in glosses)
        return self._chat("Turn these Indian Sign Language glosses into one natural English question. "
                          f"Output only the sentence.\nGlosses: {' '.join(glosses)}", max_tokens=60)


def _numbers(text: str) -> set[str]:
    return set(re.findall(r"\d+(?:[.,]\d+)*", text))

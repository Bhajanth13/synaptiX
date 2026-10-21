"""Small language model wrapper (llama.cpp, CPU-only, 4-bit GGUF).

The SLM never supplies facts. It only rewrites verified text into plain language, or turns
sign-language glosses into a sentence. Safeguards:
- the user's question is never sent to the model (it only sees the verified text and the term);
- every rewrite goes through `guard.check_rewrite`; a failing rewrite is replaced by the source;
- rewrites can be prerendered offline (`synaptix prerender`) and shipped in data/rewrites/;
- the model is loaded lazily, so nothing slow happens unless a rewrite is actually needed.
If no model is installed, every call passes the input through unchanged.
"""

import os
from pathlib import Path

from synaptix.llm import guard, rewrites
from synaptix.llm.guard import Check
from synaptix.llm.rewrites import RewriteStore

DEFAULT_MODEL = Path(__file__).resolve().parents[2] / "models" / "slm.gguf"
PROMPT_VERSION = 2

SYSTEM = (
    "You rewrite one verified text about Indian civic or legal matters into plain English for adult "
    "readers in rural India. Use 2-4 short sentences and common words. Keep the given term exactly as "
    "written. Keep every number. Use only facts in the text: do not add examples, comparisons, "
    "definitions of other words, advice, or opinions. Respectful, neutral tone; never childish. "
    "Output only the rewrite."
)

FEW_SHOT = [
    ("Term: RTI\nText: A law that lets any citizen ask a government office for information in writing. "
     "The office must reply within 30 days. People below the poverty line pay no fee.",
     "RTI is a law. Any citizen can ask a government office for information in writing. "
     "The office must reply within 30 days. People below the poverty line pay no fee."),
    ("Term: Lok Adalat\nText: A people's court where disputes are settled by agreement. "
     "No court fee, and the decision is final, like a court order.",
     "A Lok Adalat is a people's court. Disputes are settled there by agreement. "
     "There is no court fee, and the decision is final, like a court order."),
]

GLOSS_SYSTEM = ("You turn Indian Sign Language glosses into one short, natural English question. "
                "Use only the given words. Output only the question.")
GLOSS_SHOT = ("Glosses: LAND NOTICE MEANING WHAT", "What is the meaning of a land notice?")
STOP = ["\n\nTerm:", "\n\nText:"]


class SLM:
    def __init__(self, model_path: str | None = None, n_ctx: int = 1024, store: RewriteStore | None = None):
        self._path = Path(model_path or os.environ.get("SYNAPTIX_SLM", DEFAULT_MODEL))
        self.n_ctx = n_ctx
        self.store = store if store is not None else RewriteStore()
        self._llm = None
        self._tried = False

    @property
    def llm(self):
        if not self._tried:
            self._tried = True
            if self._path.exists():
                try:
                    import llama_cpp
                    threads = int(os.environ.get("SYNAPTIX_SLM_THREADS") or max(1, (os.cpu_count() or 2) // 2))
                    self._llm = llama_cpp.Llama(model_path=str(self._path), n_ctx=self.n_ctx,
                                                n_threads=threads, verbose=False)
                    try:
                        self._llm.set_cache(llama_cpp.LlamaRAMCache(capacity_bytes=256 << 20))
                    except Exception:
                        pass
                except Exception:  # ImportError, bad file, out of memory: run without a model
                    self._llm = None
        return self._llm

    @llm.setter
    def llm(self, value):
        self._llm, self._tried = value, True

    @property
    def available(self) -> bool:
        return self.llm is not None

    def _chat(self, user, max_tokens: int = 160, system: str = SYSTEM, shots=None, stop=None) -> str:
        if isinstance(user, str):
            msgs = [{"role": "system", "content": system}]
            for u, a in (FEW_SHOT if shots is None else shots):
                msgs += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
            msgs.append({"role": "user", "content": user})
        else:
            msgs = user
        out = self.llm.create_chat_completion(messages=msgs, max_tokens=max_tokens, temperature=0.0,
                                              stop=STOP if stop is None else stop)
        return out["choices"][0]["message"]["content"].strip()

    def rewrite(self, source: str, term: str | None = None, aliases=()) -> tuple[str, Check | None, str]:
        """Return (text, check, how); how is prerendered|cache|model|fallback|passthrough."""
        stored = self.store.get(source, term)
        if stored:
            return stored, None, "prerendered"
        k = rewrites.key(source, term)
        if k in rewrites.CACHE:
            return rewrites.CACHE[k], None, "cache"
        if not self.available:
            return source, None, "passthrough"
        try:
            out = guard.clean(self._chat(f"Term: {term or 'none'}\nText: {source}"))
        except Exception:
            return source, None, "fallback"
        check = guard.check_rewrite(source, out, term, aliases)
        if check.ok:
            rewrites.CACHE[k] = out
            return out, check, "model"
        return source, check, "fallback"

    def simplify(self, verified_text: str, question: str = "", term: str | None = None, aliases=()) -> str:
        # `question` is accepted for API compatibility and deliberately never used: sending it
        # made the model substitute the user's words into the answer.
        return self.rewrite(verified_text, term, aliases)[0]

    def glosses_to_sentence(self, glosses: list[str]) -> str:
        """ISL gloss sequence (e.g. LAND NOTICE MEANING WHAT) -> English question."""
        fallback = " ".join(g.lower() for g in glosses)
        if not self.available:
            return fallback
        try:
            out = guard.clean(self._chat(f"Glosses: {' '.join(glosses)}", max_tokens=60, system=GLOSS_SYSTEM,
                                         shots=[GLOSS_SHOT], stop=["\n"]))
        except Exception:
            return fallback
        words = {guard.stem(w) for w in (g.lower() for g in glosses)}
        got = {guard.stem(w.strip("?.,")) for w in out.lower().split()}
        if len(out.split()) <= 20 and out.endswith("?") and len(words & got) >= len(words) / 2:
            return out
        return fallback

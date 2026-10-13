# SynaptiX

**Offline-first civic AI for India.** SynaptiX explains laws, turns political and legal jargon
into plain words, and tells people *who actually has the power* to fix their problem
(Sarpanch, MLA or MP), in Indian languages and Indian Sign Language.

> SynaptiX gives **legal information, not legal advice**. It never gives opinions on parties or politicians.

## What it does

| Skill | Example question | How it answers |
|---|---|---|
| **Jargon → plain words** | "What is MPLADS?", "ग्राम सभा क्या है?" | Verified glossary, rewritten simply by a small local model |
| **Who holds the power** | "Who fixes the village road?" | Deterministic lookup over the Constitution's 7th & 11th Schedules: no AI guessing |
| **Legal information** | "My contractor didn't pay minimum wage" | Expert-verified cards + what to do + who to contact (DLSA, labour office) |
| **Upcoming events** | "When is the next Gram Sabha?" | Local events feed, synced by update bundles |

## Architecture

```
INPUT                   CORE                        SKILLS                      OUTPUT
Sign (webcam) ─┐                                 ┌─ jargon        ─┐
Voice (ASR)  ──┼─▶ language detect ─▶ router ──┼─ jurisdiction  ─┼─▶ text
Text         ──┤   (pivot to English)           ├─ legal         ─┤   speech (Sarvam AI TTS,
Photo (OCR)  ──┘                                 └─ events        ─┘   cached for offline)
                                                        │
                              verified knowledge base (data/) → retriever → SLM rewrites only
```

- **Facts come only from the verified knowledge base** in `data/`. The small language model (SLM)
  rewrites and simplifies; it never invents laws, numbers or schemes.
- **Runs on a CPU.** The SLM is a 1–2B model, 4-bit GGUF via llama.cpp. Everything works even with no
  model installed (verified text is returned as-is).
- **Sign language:** MediaPipe Holistic extracts hand and body keypoints → a small classifier maps
  them to ISL glosses → the SLM turns glosses into a sentence → normal pipeline.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) and [docs/RESPONSIBLE_AI.md](docs/RESPONSIBLE_AI.md).

## Languages

Hindi, English, Bengali, Marathi, Telugu, Tamil, Gujarati, **Kannada**, Malayalam, Odia, Punjabi and
**Marwari**. Speech uses [Sarvam AI](https://www.sarvam.ai/) Bulbul voices; Marwari has no Sarvam voice
yet, so it is spoken with the Hindi voice. Run `synaptix languages` for the full list.

## Quick start

```bash
uv venv -p 3.12 && source .venv/bin/activate
uv pip install -e '.[dev]'

synaptix ask "what is zero hour"
synaptix ask "who is responsible for the national highway"
synaptix ask "ग्राम सभा क्या है"
synaptix ask "my contractor did not pay minimum wage"
synaptix signs NOTICE SUMMONS COURT MEANING

pytest
```

Optional extras:

```bash
uv pip install -e '.[slm]'    # small language model (put a GGUF at models/slm.gguf)
uv pip install -e '.[sign]'   # sign-language input (MediaPipe, OpenCV)
export SARVAM_API_KEY=...     # text to speech; add --speak to `synaptix ask`
```

## Project layout

```
synaptix/
  core/        types, language detection, router, translation, pipeline
  skills/      jargon, jurisdiction, legal, events
  kb/          knowledge-base loader + CPU retriever
  llm/         llama.cpp wrapper and guardrail prompt
  outputs/     Sarvam AI TTS with offline cache and espeak-ng fallback
  inputs/sign/ MediaPipe keypoints + sign classifier
data/          glossary.yaml, legal_cards.yaml, jurisdiction.yaml, events.json
training/sign/ record and train the ISL classifier
```

## Status

Early prototype. All knowledge-base entries are marked `review: pending` until a law student and a
native speaker verify them. Contributions of verified cards and translations are very welcome.

## License

[Apache License 2.0](LICENSE)

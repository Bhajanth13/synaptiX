# AGENTS.md — guide for AI coding agents

SynaptiX is an offline-first civic AI for rural India. It explains laws, turns political/legal
jargon into plain words, says who holds power over a problem (Sarpanch vs MLA vs MP), and lists
upcoming civic events, in Indian languages, with Indian Sign Language (ISL) input.

## Hard rules (do not break)

1. **Facts come only from `data/`.** The small language model (SLM) may rewrite or simplify verified
   text; it must never be the source of laws, numbers, schemes or dates. No match → return the
   "I don't have verified information… ask DLSA" answer (`Skill.no_match`), never a guess.
2. **Legal information, not legal advice.** High-stakes answers set `Answer.escalate_to` (DLSA, labour
   office, Protection Officer, 181/112).
3. **Political neutrality.** Explain terms, institutions and processes only. No opinions on parties,
   politicians or policies — in data, prompts or code.
4. **Offline first, CPU only.** No required network calls or GPU. Optional online features (Sarvam AI
   voice) must fall back to an offline path.
5. **Every KB entry needs a `source`.** New or edited entries keep `review: pending` until a human
   (law student / native speaker) verifies them. Never set `verified` yourself.
6. **Privacy.** Sign input stores keypoints, never video. Don't log user queries.

## How it works

```
query ─▶ core/language.detect (Unicode script) ─▶ core/translate (pivot to English, optional)
      ─▶ core/router.route (keyword rules) ─▶ skills/<skill>.answer
      ─▶ kb/store.search over data/*.yaml ─▶ verified translation or llm/slm.simplify
      ─▶ core/types.Answer{text, lang, skill, sources, verified, escalate_to, note}
      ─▶ outputs/tts (rule-based G2P + formant synth; optional Sarvam)
sign video ─▶ inputs/sign/landmarks (MediaPipe) ─▶ inputs/sign/classifier ─▶ glosses
           ─▶ Pipeline.ask_signs (SLM glosses→sentence) ─▶ normal pipeline
```

| Path | Role |
|---|---|
| `synaptix/core/pipeline.py` | `Pipeline.ask(text, lang, skill)` and `ask_signs(glosses)`; main entry |
| `synaptix/core/language.py` | `LANGUAGES` table, script detection, Sarvam locale mapping (Marwari → Hindi voice) |
| `synaptix/core/router.py` | Keyword intent rules (English + some Hindi/Kannada) |
| `synaptix/core/translate.py` | Per-entry verified translations (`plain_hi`, `summary_kn`, …); IndicTrans2 stub |
| `synaptix/kb/store.py` | Loads `data/`, dependency-free IDF token-overlap retriever, stopwords |
| `synaptix/skills/` | `jargon`, `legal`, `jurisdiction` (deterministic, no model), `events` |
| `synaptix/llm/slm.py` | llama.cpp wrapper; passthrough when no `models/slm.gguf`; guardrail system prompt |
| `synaptix/outputs/g2p.py` | Rule-based grapheme→phoneme for all Indic scripts (shared Unicode offsets) + rough English |
| `synaptix/outputs/formant.py` | Hard-coded formant synthesizer, numpy only, 16 kHz WAV |
| `synaptix/outputs/tts.py` | `RuleTTS` (default), `SarvamTTS` (optional, cached, falls back to RuleTTS) |
| `synaptix/inputs/sign/landmarks.py` | MediaPipe Tasks hand+pose → 225 floats/frame, raw 0..1 image coords; hands assigned by nearest pose wrist |
| `synaptix/inputs/sign/isl_model.py` | Pretrained 263-sign ISL recognizer (temporal CNN, numpy inference, `models/isl_include.npz`); `features()` is shared with training |
| `synaptix/inputs/sign/live.py` | Segments signs by hand presence; `recognize` (webcam), `recognize_file` (video) |
| `synaptix/inputs/sign/classifier.py` | Optional nearest-centroid classifier for your own recorded vocabulary |
| `data/` | `glossary.yaml`, `legal_cards.yaml`, `jurisdiction.yaml`, `events.json` |
| `training/sign/` | `train_include.py` (INCLUDE → `models/isl_include.npz`, needs torch), `record.py` + `train.py` for a custom vocabulary |

## Setup and commands

```bash
uv venv -p 3.12 && source .venv/bin/activate
uv pip install -e '.[dev]'          # extras: [slm] llama.cpp, [sign] mediapipe<1 + opencv
pytest -q                           # must stay green
synaptix ask "what is zero hour"
synaptix ask "ग्राम सभा क्या है" --speak
synaptix say "ಗ್ರಾಮ ಸಭೆ" --phonemes
python scripts/download_models.py   # default SLM: Qwen2.5 1.5B Instruct Q4 (~1 GB, ~6 s per answer on a laptop CPU)
synaptix sign-live                  # webcam ISL input, pretrained 263-sign model
synaptix sign-file clip.mp4 --ask   # ISL from a video file
```

Tests set `SYNAPTIX_SLM` to a missing file (`tests/conftest.py`) so they never load the real model.
`SLM.simplify` rejects any rewrite whose numbers differ from the verified text.
MediaPipe uses the Tasks API (`HandLandmarker` + `PoseLandmarker`); the legacy `mp.solutions` API is gone.

Python 3.12 (MediaPipe must stay `<1`; 1.x crashes on some recent Linux kernels).

## Common tasks

- **Add a glossary term / legal card / jurisdiction topic:** edit the YAML in `data/`, include
  `source`, `review: pending`, and keywords in English plus native script. Add a test in
  `tests/test_pipeline.py`.
- **Add a language:** add it to `LANGUAGES` in `core/language.py`; Indic scripts already work in
  `g2p.py`. Add schwa-deletion behaviour to `SCHWA_DELETING` if it applies.
- **Add a skill:** subclass `skills/base.Skill`, set `name`, implement `answer(q, english) -> Answer`,
  register it in `Pipeline.__init__`, add router keywords.
- **Swap the retriever / translator / router:** keep the same function signatures so skills don't change.

## Conventions

- Keep core dependencies to `pyyaml` and `numpy`; heavy libraries go in optional extras and are
  imported lazily.
- Every degraded path must still return an `Answer` (no crashes when models are missing).
- Match the existing style: small modules, short docstrings explaining *why*, type hints.
- Model weights live in `models/` and are git-ignored.

## Known limits

Devanagari defaults to Hindi (pass `lang` for `mr`/`mwr`). Translation backend is a stub. Router is
keyword-based. Sign recognition is isolated signs only (263-word INCLUDE vocabulary), not continuous ISL;
pause with hands down between signs. INCLUDE pose .pkl files are loaded with a restricted unpickler
(numpy arrays only) — never use plain `pickle.load` on downloaded data. English G2P is rough.

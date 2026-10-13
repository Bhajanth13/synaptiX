# Architecture

## Principles

1. **Retrieval first, generation last.** Answers come from verified data. The SLM only rewrites.
2. **Deterministic where possible.** "Who has power over X" is a lookup table, not a model.
3. **Offline first.** Everything runs locally on a CPU, including speech (hard-coded rule-based TTS).
   Sarvam AI voices are an optional online upgrade.
4. **Degrade gracefully.** No SLM → verified text as-is. No translation → English, clearly marked.
   No Sarvam → rule-based TTS. No match → "I don't know, ask DLSA", never a guess.

## Components

| Component | Today | Next |
|---|---|---|
| Language detect | Unicode script | fastText LID for Devanagari languages (hi / mr / mwr) |
| Router | Keyword rules (English + native) | Small trained intent classifier |
| Retriever | IDF token overlap, no dependencies | Multilingual embeddings (e.g. MiniLM / IndicSBERT) |
| Translation | Verified per-entry translations | IndicTrans2 distilled (offline, 22 languages) |
| SLM | llama.cpp, any 1–2B GGUF (Sarvam-1, Gemma 3 1B, Qwen 2.5 1.5B) | Fine-tune on plain-language rewrites |
| Speech in | — | AI4Bharat IndicConformer / faster-whisper |
| Speech out | Hard-coded G2P rules + formant synthesizer (offline); optional Sarvam AI | Medial schwa deletion, per-language prosody, diphone concatenation |
| Sign in | MediaPipe keypoints + nearest-centroid classifier | GRU / tiny transformer on INCLUDE + civic vocabulary |
| OCR | — | Tesseract / IndicOCR for notices and contracts |

## Sign language scope

Isolated-sign recognition over a fixed vocabulary (~100–300 civic signs plus fingerspelling) is
realistic on a CPU. Continuous ISL sentence translation is an open research problem and is not
promised. Only keypoints are stored, never video.

## Data flow

```
query ─▶ detect language ─▶ (translate to English) ─▶ route ─▶ skill
skill ─▶ search verified KB ─▶ pick verified translation or SLM-simplify ─▶ Answer
Answer{text, lang, sources, verified, escalate_to} ─▶ text / Sarvam TTS
```

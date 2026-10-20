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
| Sign in | MediaPipe Tasks keypoints + pretrained 263-sign ISL temporal CNN (INCLUDE), numpy inference | Pose-guided hand crops, more civic signs, continuous signing |
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

## Sign model results

Trained with `training/sign/train_include.py` on INCLUDE pose keypoints (3,465 train clips, 263 signs).

| Test | Result |
|---|---|
| INCLUDE test split, dataset keypoints (816 clips) | 97.5% top-1, 99.5% top-3 (3-model ensemble, final epoch, no test-set selection) |
| 28 raw INCLUDE test videos through the live pipeline, `holistic` extractor (default) | **27/28 top-1 (96%), 28/28 top-3** (only miss: FALL → IT, FALL 2nd) |
| Same 28 videos, `hands` extractor (HandLandmarker + PoseLandmarker) | 23/28 top-1 (82%) |

Reproduce: `training/sign/fetch_include_videos.py` then `training/sign/eval_videos.py`.
The Holistic extractor matters because the dataset was made with MediaPipe Holistic. The public
OpenHands INCLUDE checkpoints were also tried and score ~2% on the released pose files even with an
exact port, so SynaptiX trains its own. Note: the released INCLUDE `.pkl` files store x scaled by frame
height and y by frame width; the training loader corrects for this.

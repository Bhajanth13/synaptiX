# models/

Model weights are not committed. Put files here:

| File | What | Where to get it |
|---|---|---|
| `slm.gguf` | Small language model, 4-bit GGUF (e.g. Sarvam-1 2B, Gemma 3 1B, Qwen 2.5 1.5B) | Hugging Face, quantize with llama.cpp |
| `isl_include.npz` (committed) | Pretrained ISL recognizer, 263 signs, ~1 MB | `python training/sign/train_include.py` on the INCLUDE dataset |
| `sign_classifier.npz` | Your own civic-vocabulary sign classifier (optional) | `python training/sign/train.py` |
| `hand_landmarker.task`, `pose_landmarker_lite.task` | MediaPipe models | downloaded automatically on first use |

Override the SLM path with `SYNAPTIX_SLM=/path/to/model.gguf`.

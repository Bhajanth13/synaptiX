# models/

Model weights are not committed. Put files here:

| File | What | Where to get it |
|---|---|---|
| `slm.gguf` | Small language model, 4-bit GGUF (e.g. Sarvam-1 2B, Gemma 3 1B, Qwen 2.5 1.5B) | Hugging Face, quantize with llama.cpp |
| `sign_classifier.npz` | ISL sign classifier | `python training/sign/train.py` |

Override the SLM path with `SYNAPTIX_SLM=/path/to/model.gguf`.

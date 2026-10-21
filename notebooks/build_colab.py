"""Builds notebooks/train_isl_colab.ipynb (run: python notebooks/build_colab.py)."""
import json
from pathlib import Path

cells = []
md = lambda s: cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip()})
code = lambda s: cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip()})

md("""
# SynaptiX: train an Indian Sign Language model on a Kaggle dataset

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YajatGhule/synaptiX/blob/main/notebooks/train_isl_colab.ipynb)

Trains the same word-level sign model SynaptiX ships (an ensemble of small temporal CNNs over
MediaPipe Holistic keypoints) on **any Kaggle dataset of sign videos laid out one folder per word**:

```
<dataset>/.../<WORD>/<video>.mp4      e.g.  Police/clip01.mp4, Court/clip07.mov
```

The output `isl_custom.npz` drops straight into SynaptiX (`models/isl_include.npz`) and runs on a CPU with numpy only.

**Before you start:** Runtime → Change runtime type → **T4 GPU** (training is faster; keypoint
extraction runs on CPU either way). You need a Kaggle API token (kaggle.com → Settings → API → Create new token).
""")

code("""
# 1. Install SynaptiX (for the exact feature code used at inference) and dependencies
!git clone -q https://github.com/YajatGhule/synaptiX.git /content/synaptiX 2>/dev/null || git -C /content/synaptiX pull -q
!pip install -q "mediapipe>=0.10.30,<1" opencv-python-headless kagglehub pyyaml
import sys
sys.path.insert(0, "/content/synaptiX")
""")

md("## 2. Settings")
code("""
KAGGLE_DATASET = ""        # "owner/dataset-name" from the Kaggle URL, e.g. kaggle.com/datasets/owner/dataset-name
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".webm", ".mkv"}
MIN_CLIPS_PER_WORD = 4     # words with fewer videos are skipped (can't train and test on them)
VAL_FRACTION = 0.2         # held-out share of each word's videos
EPOCHS = 150
SEEDS = 3                  # ensemble size; 1 is faster, 3 is more accurate
SAVE_TO_DRIVE = True       # cache keypoints + model in Google Drive so a disconnect doesn't lose work
assert KAGGLE_DATASET, "Set KAGGLE_DATASET above"
""")

md("## 3. Kaggle login and download")
code("""
import os
try:  # Colab secrets (key icon on the left): add KAGGLE_USERNAME and KAGGLE_KEY
    from google.colab import userdata
    os.environ["KAGGLE_USERNAME"] = userdata.get("KAGGLE_USERNAME")
    os.environ["KAGGLE_KEY"] = userdata.get("KAGGLE_KEY")
except Exception:
    import kagglehub
    kagglehub.login()  # paste username and key when asked

import kagglehub
DATA = kagglehub.dataset_download(KAGGLE_DATASET)
print("Downloaded to", DATA)
""")

md("## 4. Find videos and words")
code("""
from collections import Counter
from pathlib import Path
from synaptix.inputs.sign.isl_model import clean_label

videos = [p for p in Path(DATA).rglob("*") if p.suffix.lower() in VIDEO_EXTS]
label_of = {v: clean_label(v.parent.name) for v in videos}
counts = Counter(label_of.values())
keep = sorted(w for w, n in counts.items() if n >= MIN_CLIPS_PER_WORD)
videos = [v for v in videos if label_of[v] in keep]
print(f"{len(videos)} videos, {len(keep)} words (skipped {len(counts) - len(keep)} words with < {MIN_CLIPS_PER_WORD} clips)")
print("Most videos:", counts.most_common(5), "\\nFewest kept:", sorted(((counts[w], w) for w in keep))[:5])
assert len(keep) >= 2, "Need at least 2 words with enough videos. Is the dataset laid out one folder per word?"
""")

md("""
## 5. Extract keypoints with MediaPipe Holistic

Uses all CPU cores. Results are cached per video, so re-running skips finished ones.
""")
code("""
import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor

CACHE = Path("/content/drive/MyDrive/synaptix_isl/keypoints" if SAVE_TO_DRIVE else "/content/keypoints")
if SAVE_TO_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
CACHE.mkdir(parents=True, exist_ok=True)

import hashlib

def cache_path(v):  # stable across processes and sessions (unlike hash())
    return CACHE / (hashlib.sha1(str(v.relative_to(DATA)).encode()).hexdigest()[:16] + ".npy")

def extract(v):
    out = cache_path(v)
    if out.exists():
        return
    import os, sys
    os.environ["GLOG_minloglevel"] = "2"
    sys.path.insert(0, "/content/synaptiX")
    from synaptix.inputs.sign.landmarks import LandmarkExtractor
    from synaptix.inputs.sign.isl_model import from_synaptix
    try:
        seq = LandmarkExtractor(backend="holistic").video(str(v), max_frames=600)
        np.save(out, from_synaptix(seq).astype(np.float32) if len(seq) else np.zeros((0, 75, 2), np.float32))
    except Exception as e:
        print("failed:", v, e)

todo = [v for v in videos if not cache_path(v).exists()]
print(f"{len(todo)} videos to process on {os.cpu_count()} cores")
with ProcessPoolExecutor(os.cpu_count()) as pool:
    for i, _ in enumerate(pool.map(extract, todo, chunksize=4), 1):
        if i % 50 == 0:
            print(i, "/", len(todo), flush=True)
""")

md("## 6. Train / validation split (stratified per word)")
code("""
rng = np.random.default_rng(0)
labels = keep
X, y = [], []
for v in videos:
    p = cache_path(v)
    if p.exists():
        k = np.load(p)
        if len(k) >= 4 and k[:, 33:].any():  # skip clips where no hands were found
            X.append(k); y.append(labels.index(label_of[v]))
y = np.array(y)
tr_idx, va_idx = [], []
for c in range(len(labels)):
    idx = rng.permutation(np.flatnonzero(y == c))
    n_va = max(1, int(round(len(idx) * VAL_FRACTION)))
    va_idx += list(idx[:n_va]); tr_idx += list(idx[n_va:])
Xtr, ytr = [X[i] for i in tr_idx], y[tr_idx]
Xva, yva = [X[i] for i in va_idx], y[va_idx]
print(f"train {len(Xtr)} clips, validation {len(Xva)} clips, {len(labels)} words")
print("Note: if the same people sign in both sets, validation accuracy will be optimistic.")
""")

md("## 7. Train the ensemble")
code("""
import torch
from synaptix.inputs.sign.isl_model import features
sys.path.insert(0, "/content/synaptiX/training/sign")
from train_include import train_one

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)
va = torch.tensor(np.stack([features(k) for k in Xva]), dtype=torch.float32)
members, probs = {}, []
for seed in range(SEEDS):
    sd, p = train_one(seed, EPOCHS, Xtr, ytr, va, yva, len(labels), device=device)
    members.update({f"m{seed}__{k}": v for k, v in sd.items()})
    probs.append(p)
    ens = np.mean(probs, axis=0)
    top1 = (ens.argmax(1) == yva).mean()
    top3 = np.mean([t in row for t, row in zip(yva, np.argsort(-ens, 1)[:, :3])])
    print(f"ensemble of {seed + 1}: validation top-1 {top1:.1%}, top-3 {top3:.1%}")
""")

md("## 8. Which words get confused?")
code("""
pred = ens.argmax(1)
wrong = Counter((labels[a], labels[b]) for a, b in zip(yva, pred) if a != b)
print("Most common mistakes (true -> predicted):")
for (t, p_), n in wrong.most_common(15):
    print(f"  {t:20} -> {p_:20} x{n}")
""")

md("## 9. Save the model")
code("""
OUT = Path("/content/drive/MyDrive/synaptix_isl" if SAVE_TO_DRIVE else "/content") / "isl_custom.npz"
np.savez_compressed(OUT, labels=np.array(labels), test_top1=top1, test_top3=top3, n_members=SEEDS, **members)
print("Saved", OUT, f"({OUT.stat().st_size / 1e6:.1f} MB)")
print("Use it: copy to synaptiX/models/isl_include.npz, or ISLModel('path/to/isl_custom.npz')")
from google.colab import files
files.download(str(OUT))
""")

md("## 10. Try it on a video (optional)")
code("""
from google.colab import files
from synaptix.inputs.sign.isl_model import ISLModel, from_synaptix
from synaptix.inputs.sign.landmarks import LandmarkExtractor

m = ISLModel(OUT)
for name in files.upload():
    seq = LandmarkExtractor(backend="holistic").video(name, 600)
    print(name, "->", m.predict_kps75(from_synaptix(seq), top=3))
""")

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
      "kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
Path(__file__).with_name("train_isl_colab.ipynb").write_text(json.dumps(nb, indent=1))
print("wrote notebooks/train_isl_colab.ipynb")

"""Train the ISL fingerspelling model (A-Z, 1-9) from an image dataset with one folder per class.

Default data: Kaggle "Indian Sign Language (ISL)" by Prathum Arikeri (CC BY-SA 4.0), 42,745 images:
    pip install kagglehub torch "mediapipe>=0.10.30,<1" opencv-python-headless
    python training/sign/train_fingerspell.py --kaggle prathumarikeri/indian-sign-language-isl
or  python training/sign/train_fingerspell.py --images /path/to/Indian

The images in each folder are consecutive video frames of one signer, so a random split would put
near-identical frames in both sets. Validation therefore holds out a contiguous block of frames per
class (--val-block), which is stricter, but it is still the same person and background: expect lower
accuracy on new signers. Writes models/isl_fingerspell.npz (numpy-only at runtime).
"""

import argparse
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from synaptix.inputs.sign.fingerspell import N_FEATURES, hand_features, mirror  # noqa: E402

OUT = ROOT / "models" / "isl_fingerspell.npz"
ROT_STD = 0.15
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _extract_chunk(paths):
    os.environ.setdefault("GLOG_minloglevel", "2")
    import cv2
    from synaptix.inputs.sign.fingerspell import HandDetector
    det = HandDetector()
    out = []
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            out.append(None)
            continue
        if max(img.shape[:2]) < 256:  # small images: upscale so the hand detector finds the hand
            img = cv2.resize(img, None, fx=256 / max(img.shape[:2]), fy=256 / max(img.shape[:2]))
        hands = det.hands(img)
        out.append(hand_features(hands) if hands else None)
    return out


def extract(paths, cache: Path):
    if cache.exists():
        d = np.load(cache, allow_pickle=False)
        return d["X"], d["ok"]
    chunks = [paths[i:i + 500] for i in range(0, len(paths), 500)]
    feats = []
    with ProcessPoolExecutor(os.cpu_count()) as pool:
        for i, res in enumerate(pool.map(_extract_chunk, chunks), 1):
            feats += res
            print(f"extracted {min(i * 500, len(paths))}/{len(paths)}", flush=True)
    ok = np.array([f is not None for f in feats])
    X = np.stack([f if f is not None else np.zeros(N_FEATURES, np.float32) for f in feats])
    np.savez_compressed(cache, X=X, ok=ok)
    return X, ok


def augment(X, rng):
    X = X.copy()
    flip = rng.random(len(X)) < 0.5
    X[flip] = np.stack([mirror(x) for x in X[flip]]) if flip.any() else X[flip]
    # small rotation + scale + jitter of each hand's normalized points
    for h in (slice(0, 42), slice(42, 84)):
        pts = X[:, h].reshape(len(X), 21, 2)
        a = rng.normal(0, ROT_STD, len(X))
        s = rng.uniform(0.9, 1.1, len(X))
        R = np.stack([np.stack([np.cos(a), -np.sin(a)], 1), np.stack([np.sin(a), np.cos(a)], 1)], 1) * s[:, None, None]
        pts = np.einsum("nij,nkj->nki", R, pts) + rng.normal(0, 0.02, pts.shape)
        X[:, h] = pts.reshape(len(X), 42) * (np.abs(X[:, h]).sum(1, keepdims=True) > 0)
    return X


def main():
    import torch
    import torch.nn as nn
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=Path, nargs="*", default=[], help="image folders (one subfolder per class)")
    ap.add_argument("--kaggle", nargs="*", default=[], help="owner/dataset slugs, downloaded with kagglehub")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--val-block", type=float, default=0.2, help="last share of each class's frames held out")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--rot", type=float, default=0.15, help="std of random rotation (radians) in augmentation")
    args = ap.parse_args()
    global ROT_STD
    ROT_STD = args.rot
    roots = list(args.images)
    if args.kaggle:
        import kagglehub
        roots += [Path(kagglehub.dataset_download(k)) for k in args.kaggle]
    num = lambda p: int(m.group()) if (m := re.search(r"\d+", p.stem)) else 0
    sets = []
    for root in roots:
        files = [p for p in root.rglob("*") if p.suffix.lower() in EXTS]
        files.sort(key=lambda p: (p.parent.name, num(p)))
        X, ok = extract(files, ROOT / "models" / f"fingerspell_features_{root.parent.name}_{root.name}_{len(files)}.npz")
        print(f"{root}: {len(files)} images, hands found in {ok.mean():.1%}")
        sets.append((files, X, ok))
    labels = sorted({p.parent.name.upper() for files, _, _ in sets for p in files})
    print(f"{len(labels)} classes: {' '.join(labels)}")
    Xs, ys, oks, vals = [], [], [], []
    for files, X, ok in sets:
        y = np.array([labels.index(p.parent.name.upper()) for p in files])
        val = np.zeros(len(files), bool)
        for c in np.unique(y):  # hold out the last block of frames of each class in each dataset
            idx = np.flatnonzero(y == c)
            val[idx[int(len(idx) * (1 - args.val_block)):]] = True
        Xs.append(X); ys.append(y); oks.append(ok); vals.append(val)
    X, y, ok, val = np.concatenate(Xs), np.concatenate(ys), np.concatenate(oks), np.concatenate(vals)
    tr, va = ok & ~val, ok & val
    Xtr, ytr, Xva, yva = X[tr], y[tr], X[va], y[va]
    print(f"train {len(Xtr)}, validation {len(Xva)}")

    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    net = nn.Sequential(nn.Linear(N_FEATURES, 256), nn.ReLU(), nn.Dropout(0.2),
                        nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, len(labels)))
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-3)
    steps = args.epochs * (len(Xtr) // 256 + 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps)
    xva = torch.tensor(Xva)
    for ep in range(args.epochs):
        net.train()
        order = rng.permutation(len(Xtr))
        for b in range(0, len(order), 256):
            i = order[b:b + 256]
            loss = nn.functional.cross_entropy(net(torch.tensor(augment(Xtr[i], rng))), torch.tensor(ytr[i]),
                                               label_smoothing=0.05)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        if ep % 10 == 9 or ep == args.epochs - 1:
            net.eval()
            with torch.no_grad():
                acc = (net(xva).argmax(1).numpy() == yva).mean()
            print(f"epoch {ep + 1}: loss {loss.item():.3f}  validation top-1 {acc:.1%}", flush=True)
    lin = [m for m in net if isinstance(m, nn.Linear)]
    weights = {}
    for i, m in enumerate(lin):
        weights[f"W{i}"] = m.weight.detach().numpy()
        weights[f"b{i}"] = m.bias.detach().numpy()
    args.out.parent.mkdir(exist_ok=True)
    np.savez_compressed(args.out, labels=np.array(labels), n_layers=len(lin), val_top1=acc, **weights)
    print(f"validation top-1 {acc:.1%} -> {args.out}")


if __name__ == "__main__":
    main()

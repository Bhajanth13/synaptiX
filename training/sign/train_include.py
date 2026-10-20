"""Train the SynaptiX ISL model on the INCLUDE dataset (263 Indian signs) and export it to numpy.

Data: INCLUDE pose keypoints from AI4Bharat OpenHands (CC BY 4.0):
    https://zenodo.org/records/6674324/files/INCLUDE.zip   (unzip, then unzip INCLUDE/Pose_Signs.zip)
    Train/test split CSVs: https://github.com/AI4Bharat/OpenHands/releases/download/checkpoints_v1/include_metadata.zip

    pip install torch numpy
    python training/sign/train_include.py --poses /path/to/Pose_Signs --splits /path/to/Train_Test_Split

Writes models/isl_include.npz, loaded by synaptix.inputs.sign.isl_model (numpy only, no torch at runtime).
Pose files are pickles: they are loaded with a restricted unpickler that only allows numpy arrays.
"""

import argparse
import csv
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from synaptix.inputs.sign.isl_model import N_IN, T, clean_label, features  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "models" / "isl_include.npz"
_ALLOWED = {("numpy.core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "_reconstruct"),
            ("numpy", "ndarray"), ("numpy", "dtype"), ("numpy.core.multiarray", "scalar"),
            ("numpy._core.multiarray", "scalar")}


class _SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) not in _ALLOWED:
            raise pickle.UnpicklingError(f"blocked {module}.{name}")
        return super().find_class(module, name)


def load_clip(path):
    with open(path, "rb") as f:
        d = _SafeUnpickler(f).load()
    k = np.asarray(d["keypoints"], dtype=np.float64)[..., :2].copy()
    h, w = d["vid_shape"][:2]
    # The released files store x scaled by the frame height and y by the width (checked against
    # MediaPipe run on the original videos), so divide the same way to get 0..1 image coords.
    k[..., 0] /= h
    k[..., 1] /= w
    return k  # (frames, 75, 2): 33 pose, 21 left hand, 21 right hand


def load_split(poses: Path, csv_path: Path, labels: list[str]):
    X, y = [], []
    for r in csv.DictReader(open(csv_path)):
        p = poses / (os.path.splitext(r["FilePath"])[0] + ".pkl")
        try:
            X.append(load_clip(p))
            y.append(labels.index(clean_label(r["Word"])))
        except Exception:
            continue
    return X, np.array(y)


_POSE_PAIRS = [(1, 4), (2, 5), (3, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16), (17, 18), (19, 20),
               (21, 22), (23, 24), (25, 26), (27, 28), (29, 30), (31, 32)]
_MIRROR = list(range(75))
for a, b in _POSE_PAIRS:
    _MIRROR[a], _MIRROR[b] = b, a
_MIRROR[33:54], _MIRROR[54:75] = list(range(54, 75)), list(range(33, 54))


def mirror(k):
    """Left-handed version of a clip: flip x and swap left/right body parts."""
    k = k[:, _MIRROR].copy()
    mask = k.any(-1)
    k[..., 0] = np.where(mask, 1 - k[..., 0], 0)
    return k


def augment(k, rng):
    """Random rotation, scale, shear, shift, speed and dropped hands: robustness to webcam differences."""
    if rng.random() < 0.3:
        k = mirror(k)
    a = rng.normal(0, 0.1)
    M = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]) @ np.array([[1, rng.normal(0, 0.1)], [0, 1]])
    M *= rng.uniform(0.85, 1.15)
    mask = k.any(-1, keepdims=True)
    k = np.where(mask, (k - 0.5) @ M.T + 0.5 + rng.normal(0, 0.02, 2), 0)
    for lo, hi in ((33, 54), (54, 75)):  # simulate a weaker hand detector (live MediaPipe Tasks)
        if rng.random() < 0.5:
            k[rng.random(len(k)) < rng.uniform(0, 0.6), lo:hi] = 0
    n = len(k)
    i0, i1 = sorted(rng.integers(0, max(1, n // 6), 2))
    return k[i0: n - i1] if n - i1 - i0 >= 4 else k


class Net(nn.Module):
    def __init__(self, n_cls, h=192):
        super().__init__()
        self.c1 = nn.Conv1d(N_IN, h, 5, padding=2)
        self.c2 = nn.Conv1d(h, h, 5, padding=2)
        self.c3 = nn.Conv1d(h, h, 3, padding=1)
        self.fc = nn.Linear(2 * h, n_cls)
        self.drop = nn.Dropout(0.3)

    def forward(self, x):  # x: (B, T, N_IN)
        x = x.transpose(1, 2)
        x = torch.relu(self.c1(x))
        x = torch.relu(self.c2(x))
        x = torch.relu(self.c3(x))
        x = torch.cat([x.mean(-1), x.amax(-1)], 1)
        return self.fc(self.drop(x))


def train_one(seed, epochs, Xtr, ytr, te, yte, n_cls):
    """Train one network; returns (state dict as numpy, test probabilities from its final epoch)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = Net(n_cls)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=epochs * (len(Xtr) // 64 + 1))
    for ep in range(epochs):
        net.train()
        order = rng.permutation(len(Xtr))
        for b in range(0, len(order), 64):
            idx = order[b: b + 64]
            xb = torch.tensor(np.stack([features(augment(Xtr[i], rng)) for i in idx]), dtype=torch.float32)
            loss = nn.functional.cross_entropy(net(xb), torch.tensor(ytr[idx]), label_smoothing=0.1)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        if ep % 10 == 9 or ep == epochs - 1:
            net.eval()
            with torch.no_grad():
                logits = net(te)
            print(f"seed {seed} epoch {ep + 1}: loss {loss.item():.3f}  test top-1 "
                  f"{(logits.argmax(1).numpy() == yte).mean():.1%}", flush=True)
    # The final epoch is kept (no picking the best epoch on the test set, so the reported score is honest).
    return ({k.replace(".", "__"): v.detach().numpy() for k, v in net.state_dict().items()},
            torch.softmax(logits, 1).numpy())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--poses", type=Path, required=True)
    ap.add_argument("--splits", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--seeds", type=int, default=3, help="ensemble size; predictions are averaged")
    args = ap.parse_args()

    labels = sorted({clean_label(r["Word"]) for r in csv.DictReader(open(args.splits / "train_include.csv"))})
    Xtr, ytr = load_split(args.poses, args.splits / "train_include.csv", labels)
    Xte, yte = load_split(args.poses, args.splits / "test_include.csv", labels)
    print(f"{len(labels)} signs, {len(Xtr)} train clips, {len(Xte)} test clips")
    te = torch.tensor(np.stack([features(k) for k in Xte]), dtype=torch.float32)

    members, probs = {}, []
    for seed in range(args.seeds):
        sd, p = train_one(seed, args.epochs, Xtr, ytr, te, yte, len(labels))
        members.update({f"m{seed}__{k}": v for k, v in sd.items()})
        probs.append(p)
        ens = np.mean(probs, axis=0)
        top1 = (ens.argmax(1) == yte).mean()
        top3 = np.mean([y in row for y, row in zip(yte, np.argsort(-ens, 1)[:, :3])])
        print(f"ensemble of {seed + 1}: test top-1 {top1:.1%}  top-3 {top3:.1%}", flush=True)
    OUT.parent.mkdir(exist_ok=True)
    np.savez_compressed(OUT, labels=np.array(labels), test_top1=top1, test_top3=top3,
                        n_members=args.seeds, **members)
    best = top1
    print(f"ensemble test top-1 {best:.1%} -> {OUT}")


if __name__ == "__main__":
    main()

"""Pretrained Indian Sign Language recognizer: 263 signs from the INCLUDE dataset, numpy-only.

Trained by training/sign/train_include.py on INCLUDE pose keypoints (AI4Bharat, CC BY 4.0) and
shipped as models/isl_include.npz (a small temporal CNN). No torch needed at runtime.

Features (shared by training and inference, so webcam input matches the training data):
27 keypoints (nose, eyes, shoulders, elbows, 10 per hand), x/y in 0..1 image coords,
centred/scaled by the clip's mean shoulder midpoint and width, resampled to T frames,
plus frame-to-frame velocity. Missing hand detections are filled from the nearest frame.
"""

import re
from pathlib import Path

import numpy as np

DEFAULT_WEIGHTS = Path(__file__).resolve().parents[3] / "models" / "isl_include.npz"
SELECT_27 = [0, 2, 5, 11, 12, 13, 14, 33, 37, 38, 41, 42, 45, 46, 49, 50, 53, 54, 58, 59, 62, 63, 66, 67, 70, 71, 74]
T = 48
N_IN = 27 * 2 * 2  # positions + velocities

# INCLUDE glosses that are compounds or annotated -> plain words for the text pipeline.
GLOSS_WORDS = {"STREETORROAD": "road", "STOREORSHOP": "shop", "TRAINSTATION": "railway station",
               "TRAINTICKET": "train ticket", "CELLPHONE": "phone", "BIGLARGE": "big", "SMALLLITTLE": "small",
               "HOWAREYOU": "how are you", "THANKYOU": "thank you", "YOU(PLURAL)": "you", "RACE(ETHNICITY)": "race",
               "SECOND(NUMBER)": "second", "EX.MONSOON": "monsoon", "GOODMORNING": "good morning",
               "GOODAFTERNOON": "good afternoon", "GOODEVENING": "good evening", "GOODNIGHT": "good night"}


def clean_label(gloss: str) -> str:
    """'4. sad' -> 'SAD', '83. big large' -> 'BIGLARGE' (matches the INCLUDE folder/CSV names)."""
    return re.sub(r"\s+", "", re.sub(r"^\d+\.\s*", "", gloss)).upper()


def gloss_to_word(gloss: str) -> str:
    return GLOSS_WORDS.get(gloss, gloss.lower())


def from_synaptix(seq: np.ndarray) -> np.ndarray:
    """SynaptiX frames (T, 225) [left hand, right hand, pose] -> (T, 75, 2) [pose, left hand, right hand]."""
    pts = seq.reshape(len(seq), 75, 3)
    return np.concatenate([pts[:, 42:], pts[:, :21], pts[:, 21:42]], axis=1)[:, :, :2]


def fill_missing_hands(kps75: np.ndarray) -> np.ndarray:
    """Frames where a hand wasn't detected copy the nearest frame where it was (detectors drop hands often)."""
    k = kps75.copy()
    for lo, hi in ((33, 54), (54, 75)):
        seen = np.flatnonzero(k[:, lo:hi].any(axis=(1, 2)))
        if len(seen) == 0:
            continue
        nearest = seen[np.abs(np.arange(len(k))[:, None] - seen[None]).argmin(axis=1)]
        k[:, lo:hi] = k[nearest, lo:hi]
    return k


def features(kps75: np.ndarray) -> np.ndarray:
    """(frames, 75, 2) image coords -> (T, N_IN)."""
    kps75 = fill_missing_hands(kps75)
    idx = np.linspace(0, len(kps75) - 1, T).round().astype(int)
    x = kps75[idx][:, SELECT_27, :2].astype(np.float64)
    mask = x.any(-1, keepdims=True)
    sh = x[:, 3:5]
    ok = mask[:, 3, 0] & mask[:, 4, 0]
    if ok.any():
        center = sh[ok].mean(axis=(0, 1))
        width = np.linalg.norm(sh[ok, 0] - sh[ok, 1], axis=-1).mean() or 1.0
        x = np.where(mask, (x - center) / width, 0)
    vel = np.diff(x, axis=0, prepend=x[:1])
    return np.concatenate([x.reshape(T, -1), vel.reshape(T, -1)], axis=1).astype(np.float32)


def _conv1d(x, w, b):
    """x: (C_in, T), w: (C_out, C_in, K) 'same' padding."""
    k = w.shape[2]
    pad = k // 2
    xp = np.pad(x, ((0, 0), (pad, pad)))
    cols = np.stack([xp[:, i: i + x.shape[1]] for i in range(k)], axis=-1)  # (C_in, T, K)
    return np.einsum("ctk,ock->ot", cols, w) + b[:, None]


class ISLModel:
    def __init__(self, weights: Path = DEFAULT_WEIGHTS):
        w = np.load(weights)
        self.w = {k: w[k] for k in w.files}
        self.labels = [str(g) for g in w["labels"]]
        self.test_top1 = float(w["test_top1"]) if "test_top1" in w.files else None

    def logits_kps75(self, kps75: np.ndarray) -> np.ndarray:
        w = self.w
        x = features(kps75).T  # (N_IN, T)
        for c in ("c1", "c2", "c3"):
            x = np.maximum(_conv1d(x, w[f"{c}__weight"], w[f"{c}__bias"]), 0)
        pooled = np.concatenate([x.mean(-1), x.max(-1)])
        return pooled @ w["fc__weight"].T + w["fc__bias"]

    def predict_kps75(self, kps75: np.ndarray, top: int = 3) -> list[tuple[str, float]]:
        z = self.logits_kps75(kps75)
        p = np.exp(z - z.max())
        p /= p.sum()
        idx = np.argsort(-p)[:top]
        return [(self.labels[i], float(p[i])) for i in idx]

    def predict(self, seq: np.ndarray, min_conf: float = 0.3) -> tuple[str | None, float]:
        """Same interface as SignClassifier: (T, 225) SynaptiX frames -> (gloss or None, confidence)."""
        label, conf = self.predict_kps75(from_synaptix(seq), top=1)[0]
        return (label if conf >= min_conf else None), conf

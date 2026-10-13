"""Isolated-sign classifier over keypoint sequences.

Baseline: resample each clip to T frames, add frame-to-frame motion, and match to per-sign
centroids (nearest centroid, cosine). Tiny, numpy-only, trains in seconds, good enough to
prove the pipeline. Replace with a small GRU/transformer once there is enough data
(INCLUDE dataset + our own civic-vocabulary recordings).
"""

from pathlib import Path

import numpy as np

T = 32


def featurize(seq: np.ndarray) -> np.ndarray:
    if len(seq) == 0:
        raise ValueError("empty sequence")
    idx = np.linspace(0, len(seq) - 1, T).round().astype(int)
    x = seq[idx]
    motion = np.diff(x, axis=0, prepend=x[:1])
    v = np.concatenate([x, motion], axis=1).reshape(-1)
    return v / (np.linalg.norm(v) or 1.0)


class SignClassifier:
    def __init__(self, labels: list[str] | None = None, centroids: np.ndarray | None = None):
        self.labels = labels or []
        self.centroids = centroids

    def fit(self, sequences: list[np.ndarray], labels: list[str]) -> "SignClassifier":
        self.labels = sorted(set(labels))
        feats = np.stack([featurize(s) for s in sequences])
        lab = np.array(labels)
        c = np.stack([feats[lab == l].mean(axis=0) for l in self.labels])
        self.centroids = c / np.linalg.norm(c, axis=1, keepdims=True)
        return self

    def predict(self, seq: np.ndarray, min_conf: float = 0.6) -> tuple[str | None, float]:
        sims = self.centroids @ featurize(seq)
        i = int(sims.argmax())
        return (self.labels[i] if sims[i] >= min_conf else None), float(sims[i])

    def save(self, path: Path):
        np.savez(path, labels=np.array(self.labels), centroids=self.centroids)

    @classmethod
    def load(cls, path: Path) -> "SignClassifier":
        d = np.load(path)
        return cls(list(d["labels"]), d["centroids"])

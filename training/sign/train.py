"""Train the sign classifier from training/sign/recordings/<GLOSS>/*.npy.

    python training/sign/train.py
"""

from pathlib import Path

import numpy as np

from synaptix.inputs.sign.classifier import SignClassifier

REC = Path(__file__).parent / "recordings"
OUT = Path(__file__).resolve().parents[2] / "models" / "sign_classifier.npz"


def main():
    seqs, labels = [], []
    for d in sorted(p for p in REC.iterdir() if p.is_dir()):
        for f in d.glob("*.npy"):
            seq = np.load(f)
            if len(seq):
                seqs.append(seq)
                labels.append(d.name)
    if not seqs:
        raise SystemExit(f"No recordings in {REC}. Run record.py first.")
    SignClassifier().fit(seqs, labels).save(OUT)
    print(f"Trained on {len(seqs)} clips, {len(set(labels))} signs -> {OUT}")


if __name__ == "__main__":
    main()

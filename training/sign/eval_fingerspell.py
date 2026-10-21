"""Accuracy of models/isl_fingerspell.npz on another image dataset (one folder per class), e.g. different signers.

    python training/sign/eval_fingerspell.py /path/to/ISL_Dataset
"""

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import cv2  # noqa: E402

from synaptix.inputs.sign.fingerspell import FingerspellModel, HandDetector  # noqa: E402


def main():
    root = Path(sys.argv[1])
    m, det = FingerspellModel(), HandDetector()
    n = ok = nohand = 0
    per = Counter()
    tot = Counter()
    wrong = Counter()
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            continue
        want = p.parent.name.upper()
        if want not in m.labels:
            continue
        img = cv2.imread(str(p))
        if img is None:
            continue
        if max(img.shape[:2]) < 256:
            f = 256 / max(img.shape[:2])
            img = cv2.resize(img, None, fx=f, fy=f)
        hands = det.hands(img)
        n += 1
        tot[want] += 1
        if not hands:
            nohand += 1
            continue
        got, _ = m.predict(hands, min_conf=0.0)
        if got == want:
            ok += 1
            per[want] += 1
        else:
            wrong[(want, got)] += 1
    print(f"{n} images: top-1 {ok}/{n} ({ok / n:.1%}); no hand found in {nohand}")
    print("per class:", " ".join(f"{c}:{per[c]}/{tot[c]}" for c in sorted(tot)))
    print("most common mistakes:", wrong.most_common(8))


if __name__ == "__main__":
    main()

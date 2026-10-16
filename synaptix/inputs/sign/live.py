"""Live webcam sign recognition: segment signs by hand presence, classify each, return glosses.

A sign starts when a hand appears and ends after `gap` frames with no hands. Press q to finish.
"""

from pathlib import Path

import numpy as np

from synaptix.inputs.sign.classifier import SignClassifier
from synaptix.inputs.sign.landmarks import LandmarkExtractor

DEFAULT_MODEL = Path(__file__).resolve().parents[3] / "models" / "sign_classifier.npz"


def _has_hand(vec: np.ndarray) -> bool:
    return bool(vec[: 42 * 3].any())


def recognize(camera=0, model: Path = DEFAULT_MODEL, gap: int = 8, min_len: int = 8,
              show: bool = True) -> list[str]:
    import cv2
    clf = SignClassifier.load(model)
    ext = LandmarkExtractor()
    cap = cv2.VideoCapture(camera)
    glosses, current, idle = [], [], 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        vec = ext.frame(img)
        if _has_hand(vec):
            current.append(vec)
            idle = 0
        elif current:
            idle += 1
            if idle >= gap:
                if len(current) >= min_len:
                    g, conf = clf.predict(np.stack(current))
                    if g:
                        glosses.append(g)
                current, idle = [], 0
        if show:
            cv2.putText(img, " ".join(glosses[-5:]) or "sign...", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)
            cv2.imshow("SynaptiX sign input (q to finish)", img)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    cap.release()
    if show:
        cv2.destroyAllWindows()
    return glosses

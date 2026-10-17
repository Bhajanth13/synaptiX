"""Live webcam sign recognition: segment signs by hand presence, classify each, return glosses.

Default recognizer: the pretrained INCLUDE model (263 ISL signs, models/isl_include.npz).
If you train your own civic vocabulary (training/sign/), pass model="custom".
A sign starts when a hand appears and ends after `gap` frames with no hands. Press q to finish.
"""

from pathlib import Path

import numpy as np

from synaptix.inputs.sign.landmarks import LandmarkExtractor

MODELS = Path(__file__).resolve().parents[3] / "models"


def load_recognizer(model: str = "include"):
    if model == "custom":
        from synaptix.inputs.sign.classifier import SignClassifier
        return SignClassifier.load(MODELS / "sign_classifier.npz")
    from synaptix.inputs.sign.isl_model import ISLModel
    return ISLModel(MODELS / "isl_include.npz")


def _has_hand(vec: np.ndarray) -> bool:
    return bool(vec[: 42 * 3].any())


def segment(frames: np.ndarray, gap: int = 8, min_len: int = 8) -> list[np.ndarray]:
    """Split a (T, 225) recording into signs at runs of >= gap hand-free frames."""
    signs, current, idle = [], [], 0
    for vec in frames:
        if _has_hand(vec):
            current.append(vec)
            idle = 0
        elif current:
            idle += 1
            if idle >= gap:
                if len(current) >= min_len:
                    signs.append(np.stack(current))
                current, idle = [], 0
    if len(current) >= min_len:
        signs.append(np.stack(current))
    return signs


def recognize_file(path, model: str = "include", whole: bool = False) -> list[tuple[str, float]]:
    """Recognise signs in a video. whole=True treats the clip as one sign (best for single-sign clips)."""
    rec, ext = load_recognizer(model), LandmarkExtractor()
    seq = ext.video(str(path), max_frames=3000)
    parts = [seq] if whole else segment(seq)
    return [rec.predict(s, min_conf=0.0) for s in parts if len(s)]


def recognize(camera=0, model: str = "include", gap: int = 8, min_len: int = 8, show: bool = True) -> list[str]:
    import cv2
    rec, ext = load_recognizer(model), LandmarkExtractor()
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
                    g, _ = rec.predict(np.stack(current))
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

"""Indian Sign Language fingerspelling: static hand shapes for letters A-Z and digits 1-9.

Trained by training/sign/train_fingerspell.py on the Kaggle "Indian Sign Language (ISL)" image
dataset (Prathum Arikeri, CC BY-SA 4.0) and shipped as models/isl_fingerspell.npz: a small MLP
over MediaPipe hand landmarks, numpy-only at runtime. Lets people spell names, places and words
the 263-word sign model doesn't know.

Features (shared by training and inference): up to two hands, ordered left-to-right in the image;
each hand is centred on its wrist and scaled by its size, so position and distance to the camera
don't matter; plus the offset between the two hands and a presence flag per hand.
"""

from pathlib import Path

import numpy as np

DEFAULT_WEIGHTS = Path(__file__).resolve().parents[3] / "models" / "isl_fingerspell.npz"
N_FEATURES = 2 * 21 * 2 + 2 + 2
_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"


def hand_features(hands: list[np.ndarray]) -> np.ndarray:
    """hands: list of (21, 2) arrays in image coords (x, y in 0..1), any order -> (N_FEATURES,)."""
    hands = sorted(hands, key=lambda h: h[:, 0].mean())[:2]
    out = np.zeros(N_FEATURES, dtype=np.float32)
    for i, h in enumerate(hands):
        rel = h[:, :2] - h[0, :2]
        scale = np.abs(rel).max() or 1.0
        out[i * 42:(i + 1) * 42] = (rel / scale).reshape(-1)
        out[86 + i] = 1.0
    if len(hands) == 2:
        size = max(np.abs(hands[0] - hands[0][0]).max(), 1e-6)
        out[84:86] = (hands[1][0, :2] - hands[0][0, :2]) / size
    return out


def mirror(feat: np.ndarray) -> np.ndarray:
    """Same features for the mirror image (left-handed signer): flip x and swap the hand order."""
    f = feat.copy()
    hands = [f[0:42].reshape(21, 2).copy(), f[42:84].reshape(21, 2).copy()]
    present = [f[86], f[87]]
    for h in hands:
        h[:, 0] *= -1
    if present[0] and present[1]:
        hands, present = hands[::-1], present[::-1]
        f[84:86] = [f[84], -f[85]]  # offset now points from the new left hand to the new right hand
    f[0:42], f[42:84] = hands[0].reshape(-1), hands[1].reshape(-1)
    f[86], f[87] = present
    return f


class HandDetector:
    """MediaPipe HandLandmarker in IMAGE mode (one still image or webcam frame at a time)."""

    def __init__(self, models_dir: Path = DEFAULT_WEIGHTS.parent, min_conf: float = 0.3):
        import urllib.request

        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision
        path = Path(models_dir) / "hand_landmarker.task"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(_MODEL_URL, path)
        self._mp = mp
        self.det = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(path)), running_mode=vision.RunningMode.IMAGE,
            num_hands=2, min_hand_detection_confidence=min_conf, min_hand_presence_confidence=min_conf))

    def hands(self, bgr) -> list[np.ndarray]:
        import cv2
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        res = self.det.detect(self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb))
        return [np.array([[p.x, p.y] for p in lms], dtype=np.float32) for lms in res.hand_landmarks]


class FingerspellModel:
    def __init__(self, weights: Path = DEFAULT_WEIGHTS):
        w = np.load(weights)
        self.labels = [str(x) for x in w["labels"]]
        self.layers = [(w[f"W{i}"], w[f"b{i}"]) for i in range(int(w["n_layers"]))]
        self.val_top1 = float(w["val_top1"]) if "val_top1" in w.files else None

    def probs(self, feat: np.ndarray) -> np.ndarray:
        x = feat.astype(np.float32)
        for i, (W, b) in enumerate(self.layers):
            x = x @ W.T + b
            if i < len(self.layers) - 1:
                x = np.maximum(x, 0)
        p = np.exp(x - x.max())
        return p / p.sum()

    def predict(self, hands: list[np.ndarray], min_conf: float = 0.6) -> tuple[str | None, float]:
        if not hands:
            return None, 0.0
        p = self.probs(hand_features(hands))
        i = int(p.argmax())
        return (self.labels[i] if p[i] >= min_conf else None), float(p[i])


class Speller:
    """Turns a stream of per-frame predictions into text: a letter is accepted once it is held
    for `hold` frames; dropping the hands for `gap` frames ends the word (adds a space)."""

    def __init__(self, hold: int = 8, gap: int = 12):
        self.hold, self.gap = hold, gap
        self.text, self._cur, self._n, self._idle, self._done = "", None, 0, 0, False

    def update(self, letter: str | None) -> str:
        if letter is None:
            self._idle += 1
            self._cur, self._n, self._done = None, 0, False
            if self._idle == self.gap and self.text and not self.text.endswith(" "):
                self.text += " "
            return self.text
        self._idle = 0
        if letter == self._cur:
            self._n += 1
        else:
            self._cur, self._n, self._done = letter, 1, False
        if self._n >= self.hold and not self._done:
            self.text += letter
            self._done = True  # change hand shape (or drop hands) to repeat a letter
        return self.text


def spell_live(camera=0, weights: Path = DEFAULT_WEIGHTS, show: bool = True) -> str:
    import cv2
    model, det, sp = FingerspellModel(weights), HandDetector(), Speller()
    cap = cv2.VideoCapture(camera)
    while True:
        ok, img = cap.read()
        if not ok:
            break
        letter, conf = model.predict(det.hands(img))
        text = sp.update(letter)
        if show:
            cv2.putText(img, f"{text}_   [{letter or '-'} {conf:.2f}]", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)
            cv2.imshow("SynaptiX fingerspelling (q to finish)", img)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    cap.release()
    if show:
        cv2.destroyAllWindows()
    return sp.text.strip()

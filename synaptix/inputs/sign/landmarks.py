"""Extract hand + upper-body keypoints from video with MediaPipe Tasks (CPU).

Each frame becomes a fixed 225-float vector: 21 left-hand + 21 right-hand + 33 pose points, (x, y, z).
Missing hands are zeros. Coordinates are normalized to the shoulders so signer position/size doesn't matter.
The two small .task models (~8 MB) are downloaded into models/ on first use.
Note: pin mediapipe<1 (1.x is killed on some recent Linux kernels).
"""

import os
import urllib.request
from pathlib import Path

import numpy as np

os.environ.setdefault("GLOG_minloglevel", "2")  # quiet MediaPipe's native logging
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

N_FEATURES = (21 + 21 + 33) * 3
MODELS_DIR = Path(__file__).resolve().parents[3] / "models"
_MODEL_URLS = {
    "hand_landmarker.task": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
    "pose_landmarker_lite.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
}


def _model(name: str) -> str:
    path = MODELS_DIR / name
    if not path.exists():
        MODELS_DIR.mkdir(exist_ok=True)
        urllib.request.urlretrieve(_MODEL_URLS[name], path)
    return str(path)


def _points(lms, n):
    if not lms:
        return np.zeros((n, 3), dtype=np.float32)
    return np.array([[p.x, p.y, p.z] for p in lms], dtype=np.float32)


def normalize(frame: np.ndarray) -> np.ndarray:
    """Center on the shoulder midpoint, scale by shoulder width."""
    pts = frame.reshape(-1, 3).copy()
    pose = pts[42:]
    l_sh, r_sh = pose[11], pose[12]
    if not l_sh.any() or not r_sh.any():
        return frame
    center = (l_sh + r_sh) / 2
    scale = np.linalg.norm(l_sh[:2] - r_sh[:2]) or 1.0
    mask = pts.any(axis=1)
    pts[mask] = (pts[mask] - center) / scale
    return pts.reshape(-1)


class LandmarkExtractor:
    def __init__(self):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        self._mp = mp
        mode = vision.RunningMode.VIDEO
        self.hands = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=_model("hand_landmarker.task")),
            running_mode=mode, num_hands=2))
        self.pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=_model("pose_landmarker_lite.task")),
            running_mode=mode))
        self._ts = 0

    def frame(self, bgr_image) -> np.ndarray:
        import cv2
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        self._ts += 33  # VIDEO mode needs increasing timestamps (ms)
        h = self.hands.detect_for_video(img, self._ts)
        p = self.pose.detect_for_video(img, self._ts)
        left = right = None
        for lms, hd in zip(h.hand_landmarks, h.handedness):
            if hd[0].category_name == "Left":
                left = lms
            else:
                right = lms
        pose = p.pose_landmarks[0] if p.pose_landmarks else None
        vec = np.concatenate([_points(left, 21), _points(right, 21), _points(pose, 33)]).reshape(-1)
        return normalize(vec)

    def video(self, path_or_cam=0, max_frames: int = 300) -> np.ndarray:
        import cv2
        cap = cv2.VideoCapture(path_or_cam)
        frames = []
        while len(frames) < max_frames:
            ok, img = cap.read()
            if not ok:
                break
            frames.append(self.frame(img))
        cap.release()
        return np.stack(frames) if frames else np.zeros((0, N_FEATURES), dtype=np.float32)

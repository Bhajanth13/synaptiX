"""Extract hand + upper-body keypoints from video with MediaPipe Holistic (CPU).

Each frame becomes a fixed 225-float vector: 21 left-hand + 21 right-hand + 33 pose points, (x, y, z).
Missing hands are zeros. Coordinates are normalized to the shoulders so signer position/size doesn't matter.
Note: pin mediapipe<1 (1.x is killed on some recent Linux kernels).
"""

import numpy as np

N_FEATURES = (21 + 21 + 33) * 3


def _points(lms, n):
    if lms is None:
        return np.zeros((n, 3), dtype=np.float32)
    return np.array([[p.x, p.y, p.z] for p in lms.landmark], dtype=np.float32)


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
    def __init__(self, model_complexity: int = 0):
        import mediapipe as mp
        self.holistic = mp.solutions.holistic.Holistic(model_complexity=model_complexity)

    def frame(self, bgr_image) -> np.ndarray:
        import cv2
        res = self.holistic.process(cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB))
        vec = np.concatenate([_points(res.left_hand_landmarks, 21), _points(res.right_hand_landmarks, 21),
                              _points(res.pose_landmarks, 33)]).reshape(-1)
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

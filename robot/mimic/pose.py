"""Camera frame -> 3D body landmarks with MediaPipe PoseLandmarker (runs on the Mac).

The model file (about 9 MB) comes from Google's MediaPipe model storage; ``ensure_model``
downloads it once to models/weights/mediapipe/ (gitignored). Needs ``pip install mediapipe`` (robot/mimic/requirements-mimic.txt).
"""

from __future__ import annotations

import os
import urllib.request
from typing import Any, Dict, Optional, Tuple

import numpy as np

from robot.mimic.retarget import LANDMARKS

MODEL_URLS = {
    "lite": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    "full": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    "heavy": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
}
# MediaPipe pose landmark indices
INDEX = {"left_shoulder": 11, "right_shoulder": 12, "left_elbow": 13, "right_elbow": 14,
         "left_wrist": 15, "right_wrist": 16, "left_hip": 23, "right_hip": 24,
         "left_knee": 25, "right_knee": 26, "left_ankle": 27, "right_ankle": 28}
# skeleton lines for the preview window
BONES = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24),
         (23, 25), (25, 27), (24, 26), (26, 28)]


def ensure_model(path: str, variant: str = "full") -> str:
    """Download the PoseLandmarker model to ``path`` if it is not there yet."""
    if os.path.isfile(path):
        return path
    url = MODEL_URLS[variant]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    print(f"downloading the pose model (once): {url}\n  -> {path}")
    tmp = path + ".part"
    urllib.request.urlretrieve(url, tmp)
    os.replace(tmp, path)
    return path


class PoseEstimator:
    """``detect(frame_bgr, t_ms)`` -> ({name: (xyz m, visibility)}, image points) or (None, None)."""

    def __init__(self, model_path: str) -> None:
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision
        self._mp = mp
        opts = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO, num_poses=1,
            min_pose_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.landmarker = vision.PoseLandmarker.create_from_options(opts)
        self._last_ms = -1

    def detect(self, frame_bgr: np.ndarray, t_ms: int) -> Tuple[Optional[Dict[str, Tuple[np.ndarray, float]]],
                                                               Optional[np.ndarray]]:
        t_ms = max(int(t_ms), self._last_ms + 1)          # VIDEO mode needs increasing timestamps
        self._last_ms = t_ms
        rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        res = self.landmarker.detect_for_video(image, t_ms)
        if not res.pose_world_landmarks:
            return None, None
        world = res.pose_world_landmarks[0]
        norm = res.pose_landmarks[0]
        lm = {}
        for name in LANDMARKS:
            p = world[INDEX[name]]
            vis = float(getattr(norm[INDEX[name]], "visibility", 1.0) or 0.0)
            lm[name] = (np.array([p.x, p.y, p.z]), vis)
        h, w = frame_bgr.shape[:2]
        pts = np.array([[q.x * w, q.y * h] for q in norm])
        return lm, pts

    def close(self) -> None:
        self.landmarker.close()


def draw(frame: Any, pts: Optional[np.ndarray], text: str) -> Any:
    """Skeleton + status text on a copy of the frame (mirrored so it feels like a mirror)."""
    import cv2
    img = frame.copy()
    if pts is not None:
        for a, b in BONES:
            cv2.line(img, tuple(int(v) for v in pts[a]), tuple(int(v) for v in pts[b]), (0, 220, 255), 3)
    img = cv2.flip(img, 1)
    cv2.putText(img, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 255, 40), 2)
    return img

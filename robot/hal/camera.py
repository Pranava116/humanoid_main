"""Camera capture for the real robot: picamera2 (Pi camera) or OpenCV (USB).

A background thread keeps only the latest frame so ``latest()`` never blocks
the 50 Hz control loop.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, Optional

import numpy as np

log = logging.getLogger(__name__)


class _Source:
    def read(self) -> Optional[np.ndarray]:
        raise NotImplementedError

    def close(self) -> None:
        pass


class _PiCamera(_Source):
    def __init__(self, width: int, height: int) -> None:
        from picamera2 import Picamera2  # type: ignore
        self.cam = Picamera2()
        self.cam.configure(self.cam.create_video_configuration(main={"size": (width, height), "format": "RGB888"}))
        self.cam.start()

    def read(self) -> Optional[np.ndarray]:
        # picamera2 "RGB888" is BGR in memory; flip to RGB
        return self.cam.capture_array()[:, :, ::-1].copy()

    def close(self) -> None:
        self.cam.stop()
        self.cam.close()


class _OpenCV(_Source):
    def __init__(self, index: int, width: int, height: int) -> None:
        import cv2
        self.cv2 = cv2
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise RuntimeError(f"OpenCV could not open camera {index}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def read(self) -> Optional[np.ndarray]:
        ok, frame = self.cap.read()
        return self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2RGB) if ok else None

    def close(self) -> None:
        self.cap.release()


def open_source(cfg: Dict[str, Any]) -> Optional[_Source]:
    """Open the configured camera. 'auto' tries picamera2 then OpenCV. None if nothing works."""
    kind = str(cfg.get("source", "auto"))
    w, h = int(cfg.get("width", 320)), int(cfg.get("height", 240))
    tries = {"auto": ["picamera2", "opencv"], "none": []}.get(kind, [kind])
    for t in tries:
        try:
            src = _PiCamera(w, h) if t == "picamera2" else _OpenCV(int(cfg.get("index", 0)), w, h)
            log.info("camera: %s %dx%d", t, w, h)
            return src
        except Exception as e:
            log.info("camera %s not available: %s", t, e)
    return None


class Camera:
    """Threaded latest-frame grabber."""

    def __init__(self, source: _Source, fps: float = 15.0) -> None:
        self.source = source
        self.period = 1.0 / max(1.0, fps)
        self._frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="camera", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            t0 = time.monotonic()
            try:
                f = self.source.read()
            except Exception as e:
                log.warning("camera read failed: %s", e)
                f = None
            if f is not None:
                with self._lock:
                    self._frame = f
            time.sleep(max(0.0, self.period - (time.monotonic() - t0)))

    def latest(self) -> Optional[np.ndarray]:
        with self._lock:
            return self._frame

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
        self.source.close()

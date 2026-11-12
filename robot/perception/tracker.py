"""Person tracking: filters detections, computes bearing and size, and smooths observations."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from robot.perception.detector import Detection


@dataclass(frozen=True)
class PersonObservation:
    """Observation of a person from the head camera."""

    bearing: float              # rad, horizontal angle (positive = right, negative = left)
    elevation: float            # rad, vertical angle (positive = up, negative = down)
    size: float                 # normalized bbox height (0.0..1.0)
    confidence: float           # 0.0..1.0
    bbox: Tuple[float, float, float, float]  # (xmin, ymin, xmax, ymax)
    t: float                    # timestamp (seconds)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bearing": self.bearing,
            "elevation": self.elevation,
            "size": self.size,
            "confidence": self.confidence,
            "bbox": list(self.bbox),
            "t": self.t,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PersonObservation":
        bbox = tuple(float(x) for x in d.get("bbox", [0.0, 0.0, 0.0, 0.0]))
        return cls(
            bearing=float(d.get("bearing", 0.0)),
            elevation=float(d.get("elevation", 0.0)),
            size=float(d.get("size", 0.0)),
            confidence=float(d.get("confidence", 0.0)),
            bbox=bbox,  # type: ignore[arg-type]
            t=float(d.get("t", time.time())),
        )


class PersonTracker:
    """Tracks a detected person over consecutive frames and estimates bearing and size."""

    def __init__(self, fovy_deg: float = 50.0, aspect_ratio: float = 4.0 / 3.0,
                 alpha_bearing: float = 0.4, alpha_size: float = 0.4,
                 max_lost_s: float = 2.0) -> None:
        self.fov_v_rad = math.radians(fovy_deg)
        self.fov_h_rad = self.fov_v_rad * aspect_ratio
        self.alpha_bearing = alpha_bearing
        self.alpha_size = alpha_size
        self.max_lost_s = max_lost_s

        self._last_obs: Optional[PersonObservation] = None
        self._last_t: float = 0.0

    @classmethod
    def from_config(cls, cfg: Any) -> "PersonTracker":
        fovy = float(cfg.get("perception.camera.fovy_deg", 50.0) if hasattr(cfg, "get") else 50.0)
        w = float(cfg.get("perception.camera.width", 320) if hasattr(cfg, "get") else 320)
        h = float(cfg.get("perception.camera.height", 240) if hasattr(cfg, "get") else 240)
        alpha_b = float(cfg.get("perception.tracker.alpha_bearing", 0.4) if hasattr(cfg, "get") else 0.4)
        alpha_s = float(cfg.get("perception.tracker.alpha_size", 0.4) if hasattr(cfg, "get") else 0.4)
        max_lost = float(cfg.get("perception.tracker.max_lost_s", 2.0) if hasattr(cfg, "get") else 2.0)
        return cls(fovy_deg=fovy, aspect_ratio=w / max(1.0, h),
                   alpha_bearing=alpha_b, alpha_size=alpha_s, max_lost_s=max_lost)

    def reset(self) -> None:
        self._last_obs = None
        self._last_t = 0.0

    def update(self, detections: List[Detection], t: Optional[float] = None) -> Optional[PersonObservation]:
        """Update tracker with detections at time t. Returns current PersonObservation or None."""
        now = t if t is not None else time.time()

        if self._last_obs is not None and (now - self._last_t) > self.max_lost_s:
            self._last_obs = None

        if not detections:
            return None

        # Pick best detection (highest confidence, or largest area if confidences are close)
        best = max(detections, key=lambda d: d.confidence * 0.7 + d.height * 0.3)

        # Bearing: positive = target to the right of center
        raw_bearing = (best.center_x - 0.5) * self.fov_h_rad
        # Elevation: positive = target above center
        raw_elevation = (0.5 - best.center_y) * self.fov_v_rad
        raw_size = best.height

        if self._last_obs is not None:
            bearing = (self.alpha_bearing * raw_bearing +
                       (1.0 - self.alpha_bearing) * self._last_obs.bearing)
            size = (self.alpha_size * raw_size +
                    (1.0 - self.alpha_size) * self._last_obs.size)
        else:
            bearing = raw_bearing
            size = raw_size

        obs = PersonObservation(
            bearing=bearing,
            elevation=raw_elevation,
            size=size,
            confidence=best.confidence,
            bbox=best.bbox,
            t=now,
        )
        self._last_obs = obs
        self._last_t = now
        return obs

"""Person detectors: color blob (fast sim / test default), YOLO ONNX, and MobileNet-SSD."""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Detection:
    """Bounding box normalized to 0.0..1.0 in (xmin, ymin, xmax, ymax)."""

    bbox: Tuple[float, float, float, float]
    confidence: float
    label: str = "person"

    @property
    def xmin(self) -> float:
        return self.bbox[0]

    @property
    def ymin(self) -> float:
        return self.bbox[1]

    @property
    def xmax(self) -> float:
        return self.bbox[2]

    @property
    def ymax(self) -> float:
        return self.bbox[3]

    @property
    def center_x(self) -> float:
        return (self.xmin + self.xmax) * 0.5

    @property
    def center_y(self) -> float:
        return (self.ymin + self.ymax) * 0.5

    @property
    def width(self) -> float:
        return max(0.0, self.xmax - self.xmin)

    @property
    def height(self) -> float:
        return max(0.0, self.ymax - self.ymin)

    @property
    def area(self) -> float:
        return self.width * self.height


class Detector(ABC):
    """Abstract person detector."""

    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Detect persons in an RGB frame (H, W, 3) uint8. Returns list of Detection."""


class ColorBlobDetector(Detector):
    """Fast color-based detector for simulation (red shirt: rgba='0.85 0.1 0.1 1').

    Requires no downloaded model weights. Perfect for headless CI and tests.
    """

    def __init__(self, r_min: int = 140, g_max: int = 80, b_max: int = 80,
                 min_pixels: int = 30) -> None:
        self.r_min = r_min
        self.g_max = g_max
        self.b_max = b_max
        self.min_pixels = min_pixels

    def detect(self, frame: np.ndarray) -> List[Detection]:
        if frame is None or frame.ndim != 3 or frame.shape[2] < 3:
            return []
        h, w = frame.shape[:2]
        r = frame[:, :, 0]
        g = frame[:, :, 1]
        b = frame[:, :, 2]
        mask = (r >= self.r_min) & (g <= self.g_max) & (b <= self.b_max)
        ys, xs = np.where(mask)
        count = len(xs)
        if count < self.min_pixels:
            return []
        xmin = float(xs.min()) / float(w)
        xmax = float(xs.max() + 1) / float(w)
        ymin = float(ys.min()) / float(h)
        ymax = float(ys.max() + 1) / float(h)
        # Cap confidence between 0.5 and 1.0 based on pixel count
        conf = min(1.0, 0.5 + float(count) / 600.0)
        return [Detection((xmin, ymin, xmax, ymax), conf, "person")]


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float = 0.45) -> List[int]:
    """Simple IoU Non-Maximum Suppression."""
    if len(boxes) == 0:
        return []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        union = areas[i] + areas[order[1:]] - inter
        iou = inter / np.maximum(union, 1e-6)
        inds = np.where(iou <= iou_thresh)[0]
        order = order[inds + 1]
    return keep


class YoloOnnxDetector(Detector):
    """YOLOv8 ONNX detector running via onnxruntime.

    Target class 0 is 'person' in the COCO dataset.
    """

    def __init__(self, model_path: str, conf_threshold: float = 0.4,
                 input_size: Tuple[int, int] = (320, 320)) -> None:
        import onnxruntime as ort  # type: ignore

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"YOLO ONNX model not found at {model_path}. "
                                    f"Run 'python tools/download_models.py' to download.")
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.input_size = input_size
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(model_path, sess_options=opts,
                                            providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def detect(self, frame: np.ndarray) -> List[Detection]:
        if frame is None or frame.ndim != 3:
            return []
        import cv2

        orig_h, orig_w = frame.shape[:2]
        in_w, in_h = self.input_size
        resized = cv2.resize(frame, (in_w, in_h))
        # Normalize 0..1 and shape (1, 3, H, W)
        inp = (resized.astype(np.float32) / 255.0).transpose(2, 0, 1)[None, ...]
        outputs = self.session.run(None, {self.input_name: inp})
        raw = outputs[0]  # shape (1, 84, N) or (1, N, 84)

        if raw.ndim == 3 and raw.shape[1] == 84:
            preds = raw[0].transpose(1, 0)  # (N, 84)
        else:
            preds = raw[0]

        # In YOLOv8: cols 0..3 are cx, cy, w, h in model pixel coords; col 4 is person score
        boxes = preds[:, :4]
        scores = preds[:, 4]  # class 0: person
        mask = scores >= self.conf_threshold
        if not np.any(mask):
            return []

        boxes = boxes[mask]
        scores = scores[mask]

        # Convert cx, cy, w, h to normalized xmin, ymin, xmax, ymax
        cx = boxes[:, 0] / in_w
        cy = boxes[:, 1] / in_h
        bw = boxes[:, 2] / in_w
        bh = boxes[:, 3] / in_h

        x1 = np.clip(cx - bw * 0.5, 0.0, 1.0)
        y1 = np.clip(cy - bh * 0.5, 0.0, 1.0)
        x2 = np.clip(cx + bw * 0.5, 0.0, 1.0)
        y2 = np.clip(cy + bh * 0.5, 0.0, 1.0)
        norm_boxes = np.stack([x1, y1, x2, y2], axis=1)

        keep = _nms(norm_boxes, scores)
        detections = []
        for idx in keep:
            b = norm_boxes[idx]
            detections.append(Detection((float(b[0]), float(b[1]), float(b[2]), float(b[3])),
                                        float(scores[idx]), "person"))
        return detections


class MobileNetSsdDetector(Detector):
    """OpenCV DNN MobileNet-SSD detector (Caffe model)."""

    def __init__(self, prototxt: str, weights: str, conf_threshold: float = 0.4) -> None:
        import cv2

        if not os.path.exists(prototxt) or not os.path.exists(weights):
            raise FileNotFoundError(f"MobileNet-SSD files not found ({prototxt}, {weights})")
        self.net = cv2.dnn.readNetFromCaffe(prototxt, weights)
        self.conf_threshold = conf_threshold

    def detect(self, frame: np.ndarray) -> List[Detection]:
        if frame is None or frame.ndim != 3:
            return []
        import cv2

        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
        self.net.setInput(blob)
        out = self.net.forward()
        detections = []
        for i in range(out.shape[2]):
            conf = float(out[0, 0, i, 2])
            cls_id = int(out[0, 0, i, 1])
            if cls_id == 15 and conf >= self.conf_threshold:  # 15 = person in Pascal VOC
                x1 = float(np.clip(out[0, 0, i, 3], 0.0, 1.0))
                y1 = float(np.clip(out[0, 0, i, 4], 0.0, 1.0))
                x2 = float(np.clip(out[0, 0, i, 5], 0.0, 1.0))
                y2 = float(np.clip(out[0, 0, i, 6], 0.0, 1.0))
                detections.append(Detection((x1, y1, x2, y2), conf, "person"))
        return detections


def make_detector(cfg_or_dict: Any) -> Detector:
    """Create a Detector from a Config object or a dict."""
    if hasattr(cfg_or_dict, "get"):
        backend = str(cfg_or_dict.get("perception.detector.backend",
                                      cfg_or_dict.get("detector", {}).get("backend", "color_blob")))
    else:
        backend = "color_blob"

    if backend == "yolo_onnx":
        p = (cfg_or_dict.path(cfg_or_dict.get("perception.detector.yolo_model", "models/weights/yolov8n.onnx"))
             if hasattr(cfg_or_dict, "path") else "models/weights/yolov8n.onnx")
        conf = float(cfg_or_dict.get("perception.detector.confidence_threshold", 0.4)
                     if hasattr(cfg_or_dict, "get") else 0.4)
        return YoloOnnxDetector(p, conf_threshold=conf)
    elif backend == "mobilenet_ssd":
        proto = (cfg_or_dict.path(cfg_or_dict.get("perception.detector.mobilenet_proto", ""))
                 if hasattr(cfg_or_dict, "path") else "")
        weights = (cfg_or_dict.path(cfg_or_dict.get("perception.detector.mobilenet_weights", ""))
                   if hasattr(cfg_or_dict, "path") else "")
        conf = float(cfg_or_dict.get("perception.detector.confidence_threshold", 0.4)
                     if hasattr(cfg_or_dict, "get") else 0.4)
        return MobileNetSsdDetector(proto, weights, conf_threshold=conf)
    else:
        # Default: ColorBlobDetector
        cb = cfg_or_dict.get("perception.detector.color_blob", {}) if hasattr(cfg_or_dict, "get") else {}
        if not isinstance(cb, dict):
            cb = {}
        return ColorBlobDetector(
            r_min=int(cb.get("r_min", 140)),
            g_max=int(cb.get("g_max", 80)),
            b_max=int(cb.get("b_max", 80)),
            min_pixels=int(cb.get("min_pixels", 30)),
        )

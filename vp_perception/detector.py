"""Stage 3 — DETECT: YOLO11s (COCO-pretrained) at 640 input.

Runs once per view per frame. All views of one synchronized instant are sent
to the model as a single batch, which is much faster on a GPU than four
separate calls.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .config import DetectorConfig, KEEP_CLASSES


@dataclass
class Detections:
    """Detections for ONE image, in original-image pixel coordinates.

    xyxy: (N, 4) float32  [x1, y1, x2, y2]
    conf: (N,)   float32
    cls:  (N,)   float32  COCO class id
    """
    xyxy: np.ndarray
    conf: np.ndarray
    cls: np.ndarray

    def __len__(self) -> int:
        return len(self.conf)

    # --- interface ByteTrack expects (boolean indexing + xywh) -------------
    @property
    def xywh(self) -> np.ndarray:
        if len(self) == 0:
            return np.zeros((0, 4), np.float32)
        x1, y1, x2, y2 = self.xyxy.T
        return np.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], 1).astype(np.float32)

    def __getitem__(self, idx) -> "Detections":
        return Detections(self.xyxy[idx], self.conf[idx], self.cls[idx])

    @staticmethod
    def empty() -> "Detections":
        return Detections(np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, np.float32))

    def filter(self, min_conf: float = 0.0, classes=None) -> "Detections":
        m = self.conf >= min_conf
        if classes is not None:
            m &= np.isin(self.cls.astype(int), list(classes))
        return self[m]


class YoloDetector:
    def __init__(self, cfg: DetectorConfig | None = None):
        from ultralytics import YOLO  # imported lazily so the rest of the package works without it

        self.cfg = cfg or DetectorConfig()
        self.model = YOLO(self.cfg.weights)
        self.names: dict[int, str] = self.model.names
        self.last_ms: float = 0.0

    def detect_batch(self, frames: list[np.ndarray]) -> list[Detections]:
        """frames: list of BGR uint8 images (any size). Returns one Detections per frame."""
        if not frames:
            return []
        t0 = time.perf_counter()
        kw = dict(imgsz=self.cfg.imgsz, conf=self.cfg.conf, iou=self.cfg.iou, verbose=False,
                  classes=list(self.cfg.classes) if self.cfg.classes else None)
        if self.cfg.device is not None:
            kw["device"] = self.cfg.device
        if self.cfg.half:
            kw["half"] = True
        results = self.model.predict(frames, **kw)
        out = []
        for r in results:
            b = r.boxes.cpu().numpy()
            out.append(Detections(b.xyxy.astype(np.float32), b.conf.astype(np.float32), b.cls.astype(np.float32)))
        self.last_ms = (time.perf_counter() - t0) * 1000
        return out

    def detect(self, frame: np.ndarray) -> Detections:
        return self.detect_batch([frame])[0]


def class_name(cls_id: int) -> str:
    return KEEP_CLASSES.get(int(cls_id), (str(cls_id), "other"))[0]


def class_group(cls_id: int) -> str:
    return KEEP_CLASSES.get(int(cls_id), ("", "other"))[1]

"""Configuration for the VisionPriority AI perception module (Detect + Track).

Owner: Harini (23BCE2170). Every value used by the detector and tracker lives
here so the team can tune it in one place.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

# --------------------------------------------------------------------------
# Camera views (nuScenes channel names). The deck commits to these four.
# --------------------------------------------------------------------------
VIEWS: tuple[str, ...] = ("CAM_FRONT", "CAM_BACK", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT")

# Human-readable names + short prefixes used in track labels (e.g. "FL-12").
VIEW_NAMES = {
    "CAM_FRONT": "front",
    "CAM_BACK": "rear",
    "CAM_FRONT_LEFT": "front_left",
    "CAM_FRONT_RIGHT": "front_right",
    "CAM_BACK_LEFT": "rear_left",      # heavy-vehicle profiles add the two rear-side cameras
    "CAM_BACK_RIGHT": "rear_right",
}
VIEW_PREFIX = {
    "CAM_FRONT": "F",
    "CAM_BACK": "B",
    "CAM_FRONT_LEFT": "FL",
    "CAM_FRONT_RIGHT": "FR",
    "CAM_BACK_LEFT": "RL",
    "CAM_BACK_RIGHT": "RR",
}

# --------------------------------------------------------------------------
# COCO classes we keep. Everything else (traffic lights, chairs, ...) is
# dropped before tracking. class_group is a convenience for the risk layer's
# class-vulnerability feature (C); Parvati decides the actual C values.
# --------------------------------------------------------------------------
KEEP_CLASSES: dict[int, tuple[str, str]] = {
    0: ("person", "vulnerable_road_user"),
    1: ("bicycle", "vulnerable_road_user"),
    3: ("motorcycle", "vulnerable_road_user"),
    2: ("car", "vehicle"),
    5: ("bus", "vehicle"),
    6: ("train", "vehicle"),
    7: ("truck", "vehicle"),
    15: ("cat", "animal"),
    16: ("dog", "animal"),
    17: ("horse", "animal"),
    18: ("sheep", "animal"),
    19: ("cow", "animal"),
    20: ("elephant", "animal"),
    21: ("bear", "animal"),
}


@dataclass
class DetectorConfig:
    weights: str = "yolo11s.pt"      # YOLO11s, COCO-pretrained (deck Slide 3/5)
    imgsz: int = 640                 # 640 input; Ultralytics letterboxes (keeps aspect ratio)
    conf: float = 0.10               # low on purpose: ByteTrack's 2nd stage needs low-score boxes
    iou: float = 0.70                # NMS IoU
    device: str | None = None        # None = auto (GPU if available, else CPU)
    half: bool = False               # FP16 on GPU
    classes: tuple[int, ...] = tuple(sorted(KEEP_CLASSES))


@dataclass
class TrackerConfig:
    """ByteTrack parameters (Zhang et al., ECCV 2022) — Ultralytics defaults."""
    track_high_thresh: float = 0.25  # stage-1 association uses boxes >= this
    track_low_thresh: float = 0.10   # stage-2 recovers boxes in [low, high) -> occlusion continuity
    new_track_thresh: float = 0.25   # unmatched box must be >= this to start a new ID
    track_buffer: int = 30           # frames a lost track is kept (30 @12fps = 2.5 s)
    match_thresh: float = 0.80       # association cost threshold
    fuse_score: bool = True
    min_hits_to_emit: int = 1        # ByteTrack already confirms tracks; raise to suppress 1-frame flicker
    history_len: int = 30            # how many past centroid/area samples each record carries
    class_vote: bool = True          # report majority-vote class over the track's life (stops car<->truck flicker)


@dataclass
class PerceptionConfig:
    views: tuple[str, ...] = VIEWS
    fps: float = 12.0                # nuScenes camera rate (sweeps)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)

    def to_dict(self) -> dict:
        return asdict(self)

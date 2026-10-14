"""VisionPriority AI — perception module (Stages 1-4: Input, Preprocess, Detect, Track).

Owner: Harini (23BCE2170). Produces Output Contract records consumed by the
risk / prioritisation / explanation layer (Parvati).
"""
from .config import PerceptionConfig, DetectorConfig, TrackerConfig, VIEWS
from .contract import CONTRACT_FIELDS, EXTRA_FIELDS, TrackRecord, validate_record
from .frame_source import SyncedFrameBatch, MultiViewVideoSource, SingleVideoSource, find_scene_videos

__all__ = [
    "PerceptionConfig", "DetectorConfig", "TrackerConfig", "VIEWS",
    "CONTRACT_FIELDS", "EXTRA_FIELDS", "TrackRecord", "validate_record",
    "SyncedFrameBatch", "MultiViewVideoSource", "SingleVideoSource", "find_scene_videos",
    "PerceptionPipeline", "FrameResult",
]


def __getattr__(name):  # lazy: importing the package does not load torch/ultralytics
    if name in ("PerceptionPipeline", "FrameResult"):
        from . import pipeline
        return getattr(pipeline, name)
    raise AttributeError(name)

"""Stages 1-4 glued together: SyncedFrameBatch -> list of contract records.

    from vp_perception import PerceptionPipeline, MultiViewVideoSource, find_scene_videos
    pipe = PerceptionPipeline()
    src  = MultiViewVideoSource(find_scene_videos("nuscenes_videos", "scene00"))
    for batch in src:
        result = pipe.process(batch)          # FrameResult
        for rec in result.tracks:             # dicts that follow the Output Contract
            ...                               # -> Parvati's risk layer
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from .config import PerceptionConfig
from .detector import YoloDetector
from .frame_source import SyncedFrameBatch
from .tracker import ViewTracker


@dataclass
class FrameResult:
    frame_idx: int
    timestamp: float
    tracks: list[dict]                       # Output Contract records, all views
    timing_ms: dict = field(default_factory=dict)
    source_index: dict = field(default_factory=dict)
    sync_mode: str = ""

    def to_dict(self) -> dict:
        return {"frame_idx": self.frame_idx, "timestamp": self.timestamp, "sync_mode": self.sync_mode,
                "source_index": self.source_index, "timing_ms": self.timing_ms, "tracks": self.tracks}


class PerceptionPipeline:
    def __init__(self, cfg: PerceptionConfig | None = None, views: list[str] | None = None):
        self.cfg = cfg or PerceptionConfig()
        self.detector = YoloDetector(self.cfg.detector)
        self.views = list(views or self.cfg.views)
        self.trackers = {v: ViewTracker(v, self.cfg.tracker, self.cfg.fps) for v in self.views}

    def reset(self):
        self.trackers = {v: ViewTracker(v, self.cfg.tracker, self.cfg.fps) for v in self.views}

    def process(self, batch: SyncedFrameBatch) -> FrameResult:
        t0 = time.perf_counter()
        views = [v for v in self.views if v in batch.frames]
        dets = self.detector.detect_batch([batch.frames[v] for v in views])   # one forward pass, all views
        t1 = time.perf_counter()
        tracks = []
        for v, d in zip(views, dets):
            h, w = batch.frames[v].shape[:2]
            for rec in self.trackers[v].update(d, batch.frame_idx, batch.timestamp, (w, h)):
                tracks.append(rec.to_dict())
        t2 = time.perf_counter()
        timing = {"detect": round((t1 - t0) * 1000, 2), "track": round((t2 - t1) * 1000, 2),
                  "total": round((t2 - t0) * 1000, 2), "n_detections": int(sum(len(d) for d in dets))}
        return FrameResult(batch.frame_idx, batch.timestamp, tracks, timing,
                           dict(batch.source_index), batch.sync_mode)

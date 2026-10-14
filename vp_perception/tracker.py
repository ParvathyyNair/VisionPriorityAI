"""Stage 4 — TRACK: one ByteTrack instance per camera view.

Cross-camera re-identification is out of scope (deck Slide 6/10), so the same
car seen by CAM_FRONT and CAM_FRONT_LEFT gets two independent IDs.

On top of ByteTrack this keeps a small per-track history so every emitted
record carries what the risk layer needs: centroid, bbox area, track age, hits,
and recent centroid/area samples (for approach rate A and direction D).
"""
from __future__ import annotations

from argparse import Namespace
from collections import Counter, deque
from dataclasses import dataclass, field

import numpy as np

from .config import TrackerConfig, VIEW_NAMES, VIEW_PREFIX
from .contract import TrackRecord
from .detector import Detections, class_group, class_name


@dataclass
class _TrackState:
    first_frame_idx: int
    hits: int = 0
    last_frame_idx: int = -1
    class_votes: Counter = field(default_factory=Counter)
    centroids: deque = field(default_factory=deque)
    areas: deque = field(default_factory=deque)


class ViewTracker:
    def __init__(self, view: str, cfg: TrackerConfig | None = None, fps: float = 12.0):
        from ultralytics.trackers.byte_tracker import BYTETracker

        self.view = view
        self.cfg = cfg or TrackerConfig()
        self.fps = fps
        args = Namespace(
            track_high_thresh=self.cfg.track_high_thresh,
            track_low_thresh=self.cfg.track_low_thresh,
            new_track_thresh=self.cfg.new_track_thresh,
            track_buffer=self.cfg.track_buffer,
            match_thresh=self.cfg.match_thresh,
            fuse_score=self.cfg.fuse_score,
        )
        self.bt = BYTETracker(args)
        self.states: dict[int, _TrackState] = {}

    def update(self, dets: Detections, frame_idx: int, timestamp: float,
               image_size: tuple[int, int]) -> list[TrackRecord]:
        """dets: this view's detections for this frame. image_size: (width, height)."""
        w, h = image_size
        out = self.bt.update(dets)  # rows: x1,y1,x2,y2,track_id,score,cls,det_idx
        records = []
        for row in out:
            x1, y1, x2, y2 = (round(float(v), 2) for v in np.clip(row[:4], [0, 0, 0, 0], [w, h, w, h]))
            if x2 <= x1 or y2 <= y1:
                continue
            tid, score, cls_id = int(row[4]), float(row[5]), int(row[6])
            st = self.states.get(tid)
            if st is None:
                st = self.states[tid] = _TrackState(first_frame_idx=frame_idx,
                                                    centroids=deque(maxlen=self.cfg.history_len),
                                                    areas=deque(maxlen=self.cfg.history_len))
            st.hits += 1
            st.last_frame_idx = frame_idx
            st.class_votes[cls_id] += 1
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            area = (x2 - x1) * (y2 - y1)
            st.centroids.append([round(timestamp, 4), round(cx, 2), round(cy, 2)])
            st.areas.append([round(timestamp, 4), round(area, 1)])
            if st.hits < self.cfg.min_hits_to_emit:
                continue
            vote_cls = st.class_votes.most_common(1)[0][0] if self.cfg.class_vote else cls_id
            age = frame_idx - st.first_frame_idx + 1
            records.append(TrackRecord(
                view=self.view,
                track_id=tid,
                cls=class_name(vote_cls),
                bbox=[x1, y1, x2, y2],
                confidence=round(min(max(score, 0.0), 1.0), 4),
                centroid=[round(cx, 2), round(cy, 2)],
                bbox_area=round(area, 1),
                track_age=age,
                frame_idx=frame_idx,
                timestamp=round(timestamp, 4),
                view_name=VIEW_NAMES.get(self.view, self.view),
                track_label=f"{VIEW_PREFIX.get(self.view, self.view)}-{tid:02d}",
                class_id=int(vote_cls),
                class_group=class_group(vote_cls),
                image_size=[int(w), int(h)],
                bbox_area_norm=round(area / float(w * h), 6),
                track_age_s=round(age / self.fps, 3),
                hits=st.hits,
                first_frame_idx=st.first_frame_idx,
                centroid_history=list(st.centroids),
                area_history=list(st.areas),
            ))
        # forget tracks ByteTrack itself has given up on
        stale = [t for t, s in self.states.items() if frame_idx - s.last_frame_idx > self.cfg.track_buffer + 1]
        for t in stale:
            del self.states[t]
        return records

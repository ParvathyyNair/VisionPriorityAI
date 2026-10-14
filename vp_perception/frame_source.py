"""Stage 1 — INPUT: synchronized multi-view frame batches.

This is the interface between Aishwarya's dataset pipeline and the detector.
The detector only ever sees a `SyncedFrameBatch`. Any loader (videos, image
folders, nuscenes-devkit) works as long as it yields these.

`MultiViewVideoSource` reads Aishwarya's per-camera .mp4 files
(e.g. scene00_CAM_FRONT.mp4). Sync modes:

* "timestamps" (exact): used automatically when every video has a sidecar
  `<video stem>_timestamps.csv` with columns `frame_idx,timestamp_us`
  (the nuScenes sample_data timestamp of each frame). The front camera is the
  master clock; each other view takes its nearest frame within `max_skew_s`.
* "proportional" (fallback): the videos we received have slightly different
  frame counts per view and no timestamps, so view v uses frame
  round(i * (N_v-1)/(N_master-1)). Worst-case misalignment is about half the
  frame-count difference (scene00: ~4 frames = 0.33 s at 12 fps).
"""
from __future__ import annotations

import csv
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from .config import VIEWS


@dataclass
class SyncedFrameBatch:
    frame_idx: int                         # batch index (0-based), shared by all views
    timestamp: float                       # seconds since the first batch
    frames: dict[str, np.ndarray]          # view -> BGR uint8 image at native resolution
    source_index: dict[str, int] = field(default_factory=dict)  # view -> frame number in that view's video
    sync_mode: str = "proportional"
    max_skew_s: float = 0.0                # worst timestamp disagreement inside this batch (timestamps mode)


def find_scene_videos(folder: str | Path, scene: str, views=VIEWS) -> dict[str, Path]:
    folder = Path(folder)
    out = {}
    for v in views:
        p = folder / f"{scene}_{v}.mp4"
        if not p.exists():
            raise FileNotFoundError(p)
        out[v] = p
    return out


def _read_timestamps(video: Path) -> np.ndarray | None:
    p = video.with_name(video.stem + "_timestamps.csv")
    if not p.exists():
        return None
    with open(p, newline="") as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda r: int(r["frame_idx"]))
    return np.array([float(r["timestamp_us"]) / 1e6 for r in rows])


class _VideoReader:
    """Sequential reader that can step forward to any later frame index."""

    def __init__(self, path: Path):
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise IOError(f"cannot open {path}")
        self.n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.pos = -1
        self.frame = None

    def get(self, idx: int) -> np.ndarray | None:
        if idx < self.pos:  # rare; seek back
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            self.pos = idx - 1
        while self.pos < idx:
            ok, f = self.cap.read()
            if not ok:
                return self.frame
            self.frame, self.pos = f, self.pos + 1
        return self.frame


class MultiViewVideoSource:
    def __init__(self, videos: dict[str, str | Path], fps: float = 12.0, master: str = "CAM_FRONT",
                 sync: str = "auto", max_skew_s: float = 0.05, max_frames: int | None = None):
        self.videos = {v: Path(p) for v, p in videos.items()}
        self.views = list(self.videos)
        self.master = master if master in self.videos else self.views[0]
        self.fps = fps
        self.max_frames = max_frames
        self.readers = {v: _VideoReader(p) for v, p in self.videos.items()}
        self.counts = {v: r.n for v, r in self.readers.items()}
        ts = {v: _read_timestamps(p) for v, p in self.videos.items()}
        have_ts = all(t is not None for t in ts.values())
        if sync == "auto":
            sync = "timestamps" if have_ts else "proportional"
        if sync == "timestamps" and not have_ts:
            raise ValueError("sync='timestamps' needs <video>_timestamps.csv next to every video")
        self.sync = sync
        self.ts = ts
        self.max_skew_s = max_skew_s
        if sync == "proportional" and len(set(self.counts.values())) > 1:
            warnings.warn(f"Views have different frame counts {self.counts} and no timestamps; "
                          "using proportional alignment (approximate).")
        self._plan = self._build_plan()

    def _build_plan(self) -> list[tuple[float, dict[str, int], float]]:
        nm = self.counts[self.master]
        plan = []
        if self.sync == "timestamps":
            t0 = self.ts[self.master][0]
            for i, tm in enumerate(self.ts[self.master]):
                idx, skew = {}, 0.0
                for v in self.views:
                    j = int(np.argmin(np.abs(self.ts[v] - tm)))
                    skew = max(skew, abs(self.ts[v][j] - tm))
                    idx[v] = j
                if skew <= self.max_skew_s:
                    plan.append((tm - t0, idx, skew))
        elif self.sync == "proportional":
            for i in range(nm):
                idx = {v: int(round(i * (self.counts[v] - 1) / max(nm - 1, 1))) for v in self.views}
                plan.append((i / self.fps, idx, 0.0))
        elif self.sync == "index":  # naive: same frame number everywhere, stop at the shortest video
            for i in range(min(self.counts.values())):
                plan.append((i / self.fps, {v: i for v in self.views}, 0.0))
        else:
            raise ValueError(sync)
        if self.max_frames:
            plan = plan[: self.max_frames]
        return plan

    def __len__(self) -> int:
        return len(self._plan)

    def __iter__(self) -> Iterator[SyncedFrameBatch]:
        for k, (t, idx, skew) in enumerate(self._plan):
            frames = {v: self.readers[v].get(j) for v, j in idx.items()}
            if any(f is None for f in frames.values()):
                break
            yield SyncedFrameBatch(k, round(t, 4), frames, idx, self.sync, skew)


class SingleVideoSource:
    """One video as a one-view source (used for MOT17 / BDD100K clips)."""

    def __init__(self, path: str | Path, view: str = "CAM_FRONT", fps: float | None = None):
        self.path = Path(path)
        self.view = view
        cap = cv2.VideoCapture(str(self.path))
        self.fps = fps or cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

    def __len__(self):
        return self.n

    def __iter__(self) -> Iterator[SyncedFrameBatch]:
        cap = cv2.VideoCapture(str(self.path))
        i = 0
        while True:
            ok, f = cap.read()
            if not ok:
                break
            yield SyncedFrameBatch(i, round(i / self.fps, 4), {self.view: f}, {self.view: i}, "single")
            i += 1

"""The Output Contract between Harini's perception module and Parvati's risk layer.

Deck Slide 6: view • track_id • class • bbox • confidence • centroid • bbox area • track age
              -> risk factors -> priority + explanation

`CONTRACT_FIELDS` are exactly the eight fields on the slide, with exactly these
names. Everything in `EXTRA_FIELDS` is additional context the risk layer will
need (image size for zone mapping, histories for approach rate / direction,
timestamps for rates per second). Parvati: build against these names; if one
must change, change it here and in docs/HANDOFF.md together.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field

CONTRACT_FIELDS = (
    "view",        # str   nuScenes channel, e.g. "CAM_FRONT_LEFT"
    "track_id",    # int   ByteTrack ID, unique *within its view* (not across views)
    "class",       # str   COCO class name, e.g. "person"
    "bbox",        # [x1, y1, x2, y2] float, original-image pixels (1600x900 for nuScenes)
    "confidence",  # float 0-1, detector score of the box matched this frame
    "centroid",    # [cx, cy] float, pixels
    "bbox_area",   # float, pixels^2
    "track_age",   # int, frames since this ID was first seen (1 on first frame)
)

EXTRA_FIELDS = (
    "frame_idx",        # int   index of the synchronized batch (same for all 4 views of one instant)
    "timestamp",        # float seconds since scene start (from the frame source)
    "view_name",        # str   "front" | "rear" | "front_left" | "front_right"
    "track_label",      # str   display label, e.g. "FL-12" (view prefix + id)
    "class_id",         # int   COCO id
    "class_group",      # str   "vulnerable_road_user" | "vehicle" | "animal"
    "image_size",       # [width, height] of the source frame -> normalise for zone mapping
    "bbox_area_norm",   # float bbox_area / (width*height), resolution-independent size proxy
    "track_age_s",      # float seconds since first seen
    "hits",             # int   frames in which this ID was actually matched to a detection
    "first_frame_idx",  # int
    "centroid_history", # [[t, cx, cy], ...] oldest -> newest, includes current frame
    "area_history",     # [[t, area], ...]   oldest -> newest, includes current frame
)


@dataclass
class TrackRecord:
    view: str
    track_id: int
    cls: str                      # serialised as "class" (reserved word in Python)
    bbox: list
    confidence: float
    centroid: list
    bbox_area: float
    track_age: int
    frame_idx: int
    timestamp: float
    view_name: str
    track_label: str
    class_id: int
    class_group: str
    image_size: list
    bbox_area_norm: float
    track_age_s: float
    hits: int
    first_frame_idx: int
    centroid_history: list = field(default_factory=list)
    area_history: list = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["class"] = d.pop("cls")
        # put contract fields first, in slide order, then extras
        return {k: d[k] for k in CONTRACT_FIELDS + EXTRA_FIELDS}


def validate_record(d: dict) -> list[str]:
    """Return a list of problems with one serialised record (empty list = valid)."""
    errs = []
    for k in CONTRACT_FIELDS + EXTRA_FIELDS:
        if k not in d:
            errs.append(f"missing field '{k}'")
    if errs:
        return errs
    x1, y1, x2, y2 = d["bbox"]
    w, h = d["image_size"]
    if not (x2 > x1 and y2 > y1):
        errs.append(f"degenerate bbox {d['bbox']}")
    if not (-1 <= x1 and -1 <= y1 and x2 <= w + 1 and y2 <= h + 1):
        errs.append(f"bbox {d['bbox']} outside image {w}x{h}")
    if not 0 <= d["confidence"] <= 1:
        errs.append(f"confidence {d['confidence']} not in [0,1]")
    cx, cy = d["centroid"]
    if abs(cx - (x1 + x2) / 2) > 0.51 or abs(cy - (y1 + y2) / 2) > 0.51:
        errs.append("centroid is not the bbox centre")
    if abs(d["bbox_area"] - (x2 - x1) * (y2 - y1)) > 1.0:
        errs.append("bbox_area != (x2-x1)*(y2-y1)")
    if d["track_age"] < 1 or d["hits"] < 1 or d["hits"] > d["track_age"]:
        errs.append(f"bad track_age/hits {d['track_age']}/{d['hits']}")
    if d["track_age"] != d["frame_idx"] - d["first_frame_idx"] + 1:
        errs.append("track_age != frame_idx - first_frame_idx + 1")
    if not d["area_history"] or abs(d["area_history"][-1][1] - d["bbox_area"]) > 1.0:
        errs.append("area_history must end with the current bbox_area")
    return errs

"""Unit tests that run without model weights or video files.

  python -m pytest tests -q        (or)        python tests/test_perception.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vp_perception.config import TrackerConfig  # noqa: E402
from vp_perception.contract import CONTRACT_FIELDS, validate_record  # noqa: E402
from vp_perception.detector import Detections  # noqa: E402
from vp_perception.tracker import ViewTracker  # noqa: E402


def _dets(boxes, confs, cls=0):
    if not boxes:
        return Detections.empty()
    return Detections(np.array(boxes, np.float32), np.array(confs, np.float32), np.full(len(boxes), cls, np.float32))


def test_contract_field_names_match_slide():
    assert CONTRACT_FIELDS == ("view", "track_id", "class", "bbox", "confidence", "centroid", "bbox_area", "track_age")


def test_records_are_valid_and_age_counts_up():
    tr = ViewTracker("CAM_FRONT", TrackerConfig(), fps=12)
    last = None
    for f in range(10):
        x = 100 + 10 * f
        recs = tr.update(_dets([[x, 200, x + 80, 380]], [0.9]), f, f / 12, (1600, 900))
        assert len(recs) == 1
        d = recs[0].to_dict()
        assert validate_record(d) == [], validate_record(d)
        assert list(d)[:8] == list(CONTRACT_FIELDS)
        assert d["track_age"] == f + 1 and d["hits"] == f + 1
        if last is not None:
            assert d["track_id"] == last
        last = d["track_id"]
    assert d["track_label"] == f"F-{last:02d}"
    assert len(d["area_history"]) == 10 and d["area_history"][-1][1] == d["bbox_area"]


def test_low_confidence_second_stage_keeps_id_through_partial_occlusion():
    """Object is confidently detected, then only weakly (0.15) for 5 frames, then confidently again.
    Two-stage ByteTrack should keep ONE id; it is the deck's reason for choosing ByteTrack."""
    tr = ViewTracker("CAM_FRONT_LEFT", TrackerConfig(), fps=12)
    ids, emitted = set(), 0
    for f in range(20):
        x = 300 + 6 * f
        conf = 0.15 if 8 <= f < 13 else 0.85
        recs = tr.update(_dets([[x, 300, x + 60, 450]], [conf]), f, f / 12, (1600, 900))
        emitted += len(recs) == 1
        for r in recs:
            ids.add(r.track_id)
            assert r.track_age == f - r.first_frame_idx + 1
    assert len(ids) == 1, ids
    assert emitted == 20, "two-stage ByteTrack should keep reporting the object during the weak frames"
    # contrast: with the second stage off, the object disappears for those 5 frames
    tr1 = ViewTracker("CAM_FRONT_LEFT", TrackerConfig(track_low_thresh=0.25), fps=12)
    gaps = sum(len(tr1.update(_dets([[300 + 6 * f, 300, 360 + 6 * f, 450]], [0.15 if 8 <= f < 13 else 0.85]),
                              f, f / 12, (1600, 900))) == 0 for f in range(20))
    assert gaps == 5


def test_trackers_are_independent_per_view():
    a = ViewTracker("CAM_FRONT"); b = ViewTracker("CAM_BACK")
    ra = a.update(_dets([[10, 10, 50, 50]], [0.9]), 0, 0.0, (1600, 900))
    rb = b.update(_dets([[10, 10, 50, 50]], [0.9]), 0, 0.0, (1600, 900))
    assert ra[0].view == "CAM_FRONT" and rb[0].view == "CAM_BACK"


def test_validator_catches_bad_area():
    tr = ViewTracker("CAM_FRONT")
    d = tr.update(_dets([[10, 10, 50, 50]], [0.9]), 0, 0.0, (1600, 900))[0].to_dict()
    d["bbox_area"] = 1.0
    assert any("bbox_area" in e for e in validate_record(d))


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)

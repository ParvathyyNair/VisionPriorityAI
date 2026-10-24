"""Heavy-vehicle profiles: cameras, blind-spot weighting and advice."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import vehicle_profiles as vp  # noqa: E402
from risk_layer import score_frame, score_track  # noqa: E402
from vp_ui.advisory import advise  # noqa: E402
from vp_ui.payload import build_payload  # noqa: E402


def _rec(view, cls="bicycle", y2=760, cx=1100):
    return {"view": view, "class": cls, "bbox": [cx - 80, y2 - 300, cx + 80, y2], "centroid": [cx, y2 - 150],
            "image_size": [1600, 900], "track_age": 24, "track_age_s": 2.0, "hits": 24, "track_id": 7,
            "area_history": [[0, 20000], [0.5, 28000], [1.0, 40000]], "centroid_history": [[0, cx, 600], [0.5, cx, 605], [1.0, cx, 610]]}


def test_car_profile_is_identical_to_original_scoring():
    for v in vp.CAMS_4:
        r = _rec(v)
        assert score_track(r) == score_track(r, vp.PROFILES["car"])


def test_heavy_vehicles_add_two_rear_side_cameras():
    assert len(vp.PROFILES["car"].cameras) == 4
    for k in ("van", "bus", "coach", "truck"):
        p = vp.PROFILES[k]
        assert p.heavy and len(p.cameras) == 6 and {"CAM_BACK_LEFT", "CAM_BACK_RIGHT"} <= set(p.cameras)


def test_longer_vehicle_raises_zone_criticality_monotonically():
    r = _rec("CAM_BACK_LEFT", y2=640)
    z = [score_track(r, vp.PROFILES[k])["Z"] for k in ("van", "bus", "coach", "truck")]
    assert z == sorted(z) and z[-1] > z[0]
    assert all(0 <= x <= 1 for x in z)


def test_car_ignores_rear_side_cameras():
    frame = {"tracks": [_rec("CAM_FRONT"), _rec("CAM_BACK_LEFT")]}
    assert {t["view"] for t in score_frame(frame, vp.PROFILES["car"])} == {"CAM_FRONT"}
    assert {t["view"] for t in score_frame(frame, vp.PROFILES["truck"])} == {"CAM_FRONT", "CAM_BACK_LEFT"}


def test_blind_spot_zone_and_heavy_advice():
    t = score_track(_rec("CAM_BACK_LEFT"), vp.PROFILES["truck"])
    assert t["zone"] == "blind-spot"
    assert advise({**t, "priority": "HIGH / CRITICAL"}, vp.PROFILES["truck"])[0] == "BLIND SPOT LEFT"
    assert advise({**t, "view": "CAM_BACK_RIGHT", "priority": "HIGH / CRITICAL"}, vp.PROFILES["bus"])[0] == "BLIND SPOT RIGHT"
    front = {**score_track(_rec("CAM_FRONT", "person", y2=880, cx=800), vp.PROFILES["bus"]), "priority": "HIGH / CRITICAL"}
    assert advise(front, vp.PROFILES["bus"])[0] == "DO NOT MOVE OFF"
    assert advise({**t, "view": "CAM_BACK", "priority": "HIGH / CRITICAL"}, vp.PROFILES["truck"])[0] == "DO NOT REVERSE"
    # cars keep the original advice
    assert advise({**t, "view": "CAM_FRONT_LEFT", "priority": "HIGH / CRITICAL"}, vp.PROFILES["car"])[0] == "CHECK LEFT"


def test_wheel_count_selects_a_profile():
    assert vp.profile_for_wheels(4).key == "car"
    assert vp.profile_for_wheels(6, passenger=True).key == "bus"
    assert vp.profile_for_wheels(8, passenger=True).key == "coach"
    assert vp.profile_for_wheels(18).key == "truck"


def test_six_camera_payload_on_real_scene():
    sc = ROOT / "results" / "nuscenes" / "scene02"
    base = [json.loads(l) for l in open(sc / "tracks.jsonl")]
    side = [json.loads(l) for l in open(sc / "tracks_rear_sides.jsonl")]
    assert len(base) == len(side)
    frames = [{**b, "tracks": b["tracks"] + s["tracks"]} for b, s in zip(base, side)]
    p = build_payload("scene02", frames, 12.0, vp.PROFILES["bus"])
    views = {k for f in p["frames"] for k in f["v"]}
    assert {"CAM_BACK_LEFT", "CAM_BACK_RIGHT"} & views and p["vehicle"]["key"] == "bus"
    car = build_payload("scene02", base, 12.0, vp.PROFILES["car"])
    orig = build_payload("scene02", base, 12.0)
    assert [f["k"] for f in car["frames"]] == [f["k"] for f in orig["frames"]]   # car output unchanged

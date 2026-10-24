import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vp_ui.advisory import advise


def _t(view, cls, priority, A=0.8):
    return {"view": view, "class": cls, "priority": priority, "A": A, "view_display": "front"}


def test_front_high_closing_says_brake():
    tag, msg = advise(_t("CAM_FRONT", "person", "HIGH / CRITICAL"))
    assert tag == "BRAKE READY" and "Pedestrian" in msg


def test_side_views_point_to_the_right_side():
    assert advise(_t("CAM_FRONT_LEFT", "car", "HIGH / CRITICAL"))[0] == "CHECK LEFT"
    assert advise(_t("CAM_FRONT_RIGHT", "car", "HIGH / CRITICAL"))[0] == "CHECK RIGHT"


def test_low_priority_is_only_monitoring():
    assert advise(_t("CAM_BACK", "car", "LOW", A=0.0))[0] == "MONITOR"

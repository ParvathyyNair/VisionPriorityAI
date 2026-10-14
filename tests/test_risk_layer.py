import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from risk_layer import score_track, score_frame

def test_score_range_and_formula():
    rec={"view":"CAM_FRONT","view_name":"front","track_id":1,"track_label":"F-1","class":"person",
         "bbox":[100,100,300,600],"centroid":[200,350],"image_size":[1600,900],"bbox_area":100000,
         "track_age":24,"track_age_s":2.0,"hits":22,
         "area_history":[[0,10000],[0.5,20000],[1,40000]],
         "centroid_history":[[0,200,200],[0.5,200,250],[1,200,300]]}
    t=score_track(rec)
    assert all(0 <= t[k] <= 1 for k in "CZADP")
    expected=0.25*t["C"]+0.20*t["Z"]+0.30*t["A"]+0.15*t["D"]+0.10*t["P"]
    assert abs(t["risk_score"]-round(expected,4)) < 1e-4
    assert t["priority"] in {"LOW","MEDIUM","HIGH / CRITICAL"}
    assert t["explanation"] and t["display_id"] == "F-1"

def test_new_track_has_zero_motion():
    rec={"view":"CAM_FRONT","class":"car","bbox":[100,100,200,200],"centroid":[150,150],
         "image_size":[1600,900],"track_age":1,"track_age_s":1/12,"hits":1,
         "area_history":[[0,100]],"centroid_history":[[0,150,150]]}
    t=score_track(rec)
    assert t["A"] == 0 and t["D"] == 0

def test_frame_sorted_descending():
    out=score_frame({"tracks":[
        {"view":"CAM_FRONT","class":"car","bbox":[0,0,10,10],"centroid":[5,5],"image_size":[1600,900],"track_age":1,"hits":1},
        {"view":"CAM_FRONT","class":"person","bbox":[0,0,300,800],"centroid":[150,400],"image_size":[1600,900],"track_age":30,"track_age_s":2.5,"hits":30}
    ]})
    assert [x["risk_score"] for x in out] == sorted([x["risk_score"] for x in out], reverse=True)

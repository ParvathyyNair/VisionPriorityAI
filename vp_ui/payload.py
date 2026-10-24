"""Pre-score every frame once so the browser can show alerts in step with the video."""
from __future__ import annotations

from risk_layer import score_frame
from .advisory import LEVEL, advise

VIEW_KEYS = ("CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK")
MAX_TRACKS_PER_FRAME = 12
REARM_FRAMES = 12  # a track must stay out of HIGH for ~1 s before it can raise a new event


def build_payload(scene: str, frames: list[dict], fps: float) -> dict:
    out_frames: list[dict] = []
    events: list[dict] = []
    last_high: dict[tuple, int] = {}
    prev_high: set[tuple] = set()

    for i, frame in enumerate(frames):
        scored = score_frame(frame)
        views: dict[str, list] = {}
        rows = []
        high_now: set[tuple] = set()
        for t in scored:
            lvl = LEVEL.get(t["priority"], 0)
            tag, msg = advise(t)
            key = (t.get("view"), t.get("track_id"))
            v = views.get(t["view"])
            if v is None or t["risk_score"] > v[0]:
                views[t["view"]] = [t["risk_score"], lvl]
            if len(rows) < MAX_TRACKS_PER_FRAME:
                rows.append([
                    t["display_id"], t["view"], t["class"], t["risk_score"], lvl,
                    tag, msg, [t["C"], t["Z"], t["A"], t["D"], t["P"]],
                    round(float(t.get("confidence", 0.0)), 2), t["zone"],
                ])
            if lvl == 2:
                high_now.add(key)
                if key not in prev_high and i - last_high.get(key, -10**9) > REARM_FRAMES:
                    events.append({
                        "f": i, "t": round(float(frame.get("timestamp", i / fps)), 3),
                        "id": t["display_id"], "view": t["view"], "cls": t["class"],
                        "risk": t["risk_score"], "tag": tag, "msg": msg,
                    })
                last_high[key] = i
        prev_high = high_now
        out_frames.append({
            "t": round(float(frame.get("timestamp", i / fps)), 3),
            "n": len(frame.get("tracks", [])),
            "m": scored[0]["risk_score"] if scored else 0.0,
            "v": views,
            "k": rows,
            "ms": frame.get("timing_ms", {}).get("total"),
        })
    return {"scene": scene, "fps": fps, "n": len(out_frames), "thr": [0.35, 0.65],
            "weights": {"C": 0.25, "Z": 0.20, "A": 0.30, "D": 0.15, "P": 0.10},
            "frames": out_frames, "events": events}

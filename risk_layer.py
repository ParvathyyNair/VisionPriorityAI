"""VisionPriority AI risk, priority and explanation layer.

Consumes Harini's Output Contract records. All five factor scores are normalized
to [0, 1]. The risk score is a transparent heuristic, not a calibrated probability.
"""
from __future__ import annotations
import math
import numpy as np

WEIGHTS = {"C": 0.25, "Z": 0.20, "A": 0.30, "D": 0.15, "P": 0.10}
CLASS_VULNERABILITY = {
    "person": 1.0, "bicycle": 0.9, "motorcycle": 0.9,
    "car": 0.55, "bus": 0.65, "truck": 0.7, "train": 0.7,
    "cat": 0.45, "dog": 0.45, "horse": 0.5, "sheep": 0.35,
    "cow": 0.4, "elephant": 0.5, "bear": 0.5,
}
VIEW_NAMES = {
    "CAM_FRONT": "front", "CAM_BACK": "rear",
    "CAM_FRONT_LEFT": "front-left", "CAM_FRONT_RIGHT": "front-right",
    "CAM_BACK_LEFT": "rear-left", "CAM_BACK_RIGHT": "rear-right",   # heavy-vehicle cameras
}
SIDE_VIEWS = {"CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"}

def _clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, float(x)))

def _history_slope(history, value_col, window_s=1.0, log_values=False):
    """Least-squares slope over recent history; returns 0 without 3 useful samples."""
    try:
        h = np.asarray(history, dtype=float)
        if h.ndim != 2 or len(h) < 3:
            return 0.0
        h = h[np.isfinite(h[:, 0]) & np.isfinite(h[:, value_col])]
        if len(h) < 3:
            return 0.0
        h = h[h[:, 0] >= h[-1, 0] - window_s]
        if len(h) < 3 or np.ptp(h[:, 0]) < 1e-6:
            return 0.0
        x, y = h[:, 0], h[:, value_col]
        if log_values:
            y = np.log(np.maximum(y, 1.0))
        return float(np.polyfit(x, y, 1)[0])
    except (TypeError, ValueError, IndexError, np.linalg.LinAlgError):
        return 0.0

def _zone_factor(rec):
    """Visual zone proxy: bbox bottom edge + horizontal region; not metric distance."""
    bbox = rec.get("bbox", [0, 0, 0, 0])
    size = rec.get("image_size", [1600, 900])
    w, h = max(float(size[0]), 1.0), max(float(size[1]), 1.0)
    x1, y1, x2, y2 = [float(v) for v in bbox]
    bottom = _clamp(y2 / h)
    cx = _clamp(float(rec.get("centroid", [(x1+x2)/2, (y1+y2)/2])[0]) / w)
    view = rec.get("view", "")
    # Edge regions are more relevant in the side cameras; centerline is relevant in front/rear.
    if view in ("CAM_FRONT_LEFT", "CAM_BACK_LEFT"):     # vehicle's own flank is on the image's right
        lateral = 1.0 - _clamp(abs(cx - 0.68) / 0.68)
    elif view in ("CAM_FRONT_RIGHT", "CAM_BACK_RIGHT"):
        lateral = 1.0 - _clamp(abs(cx - 0.32) / 0.68)
    elif view == "CAM_BACK":
        lateral = 1.0 - _clamp(abs(cx - 0.5) / 0.5)
    else:
        lateral = 1.0 - _clamp(abs(cx - 0.5) / 0.5)
    # Nearer-looking boxes lower in the image and nearer the relevant lateral corridor are more critical.
    proximity = _clamp((bottom - 0.35) / 0.55)
    score = 0.75 * proximity + 0.25 * lateral
    zone = "critical" if score >= 0.68 else ("caution" if score >= 0.38 else "monitoring")
    return _clamp(score), zone

def _approach_factor(rec):
    """Map log bbox-area growth to [0,1]; +0.69/s means area roughly doubles/s."""
    growth = _history_slope(rec.get("area_history", []), 1, log_values=True)
    # Fewer than 3 samples naturally gives 0; receding/static objects do not get approach credit.
    return _clamp(growth / 0.69), growth

def _direction_factor(rec):
    """View-aware directional alignment heuristic based on recent centroid velocity."""
    hist = rec.get("centroid_history", [])
    vx = vy = 0.0
    try:
        h = np.asarray(hist, dtype=float)
        if h.ndim == 2 and len(h) >= 3:
            h = h[h[:, 0] >= h[-1, 0] - 1.0]
            if len(h) >= 3 and np.ptp(h[:, 0]) > 1e-6:
                vx = float(np.polyfit(h[:, 0], h[:, 1], 1)[0])
                vy = float(np.polyfit(h[:, 0], h[:, 2], 1)[0])
    except (TypeError, ValueError, IndexError, np.linalg.LinAlgError):
        pass
    view = rec.get("view", "")
    # Positive component is motion toward the vehicle/path in the corresponding view.
    # These are image-plane heuristics; area growth remains the stronger approach signal.
    if view == "CAM_FRONT":
        toward = vy
    elif view == "CAM_BACK":
        toward = -vy
    elif view in ("CAM_FRONT_LEFT", "CAM_BACK_LEFT"):
        toward = 0.65 * vx + 0.35 * vy
    elif view in ("CAM_FRONT_RIGHT", "CAM_BACK_RIGHT"):
        toward = -0.65 * vx + 0.35 * vy
    else:
        toward = 0.0
    # Saturates at ~120 px/s. No motion / insufficient history -> neutral-low score.
    return _clamp(toward / 120.0), vx, vy

def score_track(rec, profile=None):
    """Return a copy of a contract record enriched with C/Z/A/D/P and explanation.

    profile: optional vehicle_profiles.VehicleProfile. Heavy vehicles scale the zone
    factor Z by a per-camera blind-spot gain; a car (or no profile) is unchanged."""
    out = dict(rec)
    cls = str(rec.get("class", "unknown")).lower()
    C = CLASS_VULNERABILITY.get(cls, 0.45)
    Z, zone = _zone_factor(rec)
    if profile is not None and getattr(profile, "heavy", False):
        Z = _clamp(Z * profile.gain(rec.get("view", "")))
        zone = "critical" if Z >= 0.68 else ("caution" if Z >= 0.38 else "monitoring")
        if rec.get("view") in SIDE_VIEWS and Z >= 0.6:
            zone = "blind-spot"
    A, growth = _approach_factor(rec)
    D, vx, vy = _direction_factor(rec)
    age_s = float(rec.get("track_age_s", float(rec.get("track_age", 1)) / 12.0))
    hits = max(0, int(rec.get("hits", rec.get("track_age", 1))))
    age = max(1, int(rec.get("track_age", 1)))
    # Track age rewards persistence, while hit ratio discounts repeatedly lost/re-found tracks.
    P = _clamp(min(age_s / 1.0, 1.0) * (0.7 + 0.3 * _clamp(hits / age)))
    risk = _clamp(0.25*C + 0.20*Z + 0.30*A + 0.15*D + 0.10*P)
    if risk < 0.35:
        priority = "LOW"
    elif risk < 0.65:
        priority = "MEDIUM"
    else:
        priority = "HIGH / CRITICAL"
    view_name = rec.get("view_name") or VIEW_NAMES.get(rec.get("view"), str(rec.get("view", "unknown")))
    label = rec.get("track_label") or f"{view_name[:1].upper()}-{rec.get('track_id','?')}"
    # Template-based explanation uses the same factors that feed the score.
    cls_phrase = {"person":"Person", "bicycle":"Cyclist", "motorcycle":"Motorcyclist"}.get(cls, cls.capitalize())
    motion = "approaching" if A >= 0.45 else ("moving toward the vehicle path" if D >= 0.45 else ("receding" if growth < -0.15 else "nearby"))
    reason = f"{cls_phrase} {motion} from {view_name} {zone} zone"
    out.update({
        "C": round(C, 4), "Z": round(Z, 4), "A": round(A, 4), "D": round(D, 4), "P": round(P, 4),
        "area_growth_per_s": round(growth, 4), "velocity_px_s": [round(vx, 2), round(vy, 2)],
        "zone": zone, "risk_score": round(risk, 4), "priority": priority,
        "explanation": reason, "view_display": view_name, "display_id": label,
    })
    return out

def score_frame(frame, profile=None):
    """Score all current tracks in one JSONL frame and sort most dangerous first.
    With a vehicle profile, only that vehicle's cameras are scored."""
    cams = set(profile.cameras) if profile is not None else None
    tracks = [score_track(t, profile) for t in frame.get("tracks", []) if cams is None or t.get("view") in cams]
    tracks.sort(key=lambda t: t["risk_score"], reverse=True)
    return tracks

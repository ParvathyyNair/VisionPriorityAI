"""Drawing helpers: per-view track overlays and a surround mosaic.

Mosaic layout (1600x600 at default tile size):
    [ FRONT_LEFT ][   FRONT   ][ FRONT_RIGHT ]
    [  (legend)  ][   REAR    ][   (stats)   ]
"""
from __future__ import annotations

import cv2
import numpy as np

GROUP_COLOR = {  # BGR
    "vulnerable_road_user": (60, 60, 230),
    "vehicle": (230, 160, 40),
    "animal": (40, 200, 230),
    "other": (180, 180, 180),
}


def draw_tracks(img: np.ndarray, tracks: list[dict], thickness: int = 3) -> np.ndarray:
    out = img.copy()
    for t in tracks:
        x1, y1, x2, y2 = map(int, t["bbox"])
        c = GROUP_COLOR.get(t.get("class_group", "other"), GROUP_COLOR["other"])
        cv2.rectangle(out, (x1, y1), (x2, y2), c, thickness)
        label = f"{t.get('track_label', t['track_id'])} {t['class']} {t['confidence']:.2f} age{t['track_age']}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        cv2.rectangle(out, (x1, max(0, y1 - th - 10)), (x1 + tw + 6, y1), c, -1)
        cv2.putText(out, label, (x1 + 3, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def _tile(img, title, size):
    t = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    cv2.rectangle(t, (0, 0), (size[0], 26), (25, 25, 25), -1)
    cv2.putText(t, title, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return t


def surround_mosaic(frames: dict[str, np.ndarray], tracks: list[dict], header: str = "",
                    tile=(533, 300)) -> np.ndarray:
    by_view = {}
    for t in tracks:
        by_view.setdefault(t["view"], []).append(t)
    tiles = {}
    names = {"CAM_FRONT_LEFT": "FRONT-LEFT", "CAM_FRONT": "FRONT", "CAM_FRONT_RIGHT": "FRONT-RIGHT", "CAM_BACK": "REAR"}
    for v, f in frames.items():
        n = len(by_view.get(v, []))
        tiles[v] = _tile(draw_tracks(f, by_view.get(v, [])), f"{names.get(v, v)}  |  {n} tracks", tile)
    blank = np.full((tile[1], tile[0], 3), 18, np.uint8)
    top = np.hstack([tiles.get("CAM_FRONT_LEFT", blank), tiles.get("CAM_FRONT", blank), tiles.get("CAM_FRONT_RIGHT", blank)])
    legend = blank.copy()
    y = 40
    cv2.putText(legend, "VisionPriority AI - Detect + Track", (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    for g, c in GROUP_COLOR.items():
        if g == "other":
            continue
        y += 34
        cv2.rectangle(legend, (14, y - 14), (34, y + 4), c, -1)
        cv2.putText(legend, g.replace("_", " "), (44, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA)
    y += 44
    for line in header.split("\n"):
        cv2.putText(legend, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        y += 24
    stats = blank.copy()
    counts = {}
    for t in tracks:
        counts[t["class"]] = counts.get(t["class"], 0) + 1
    y = 40
    cv2.putText(stats, f"Active tracks: {len(tracks)}", (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    for k, n in sorted(counts.items(), key=lambda kv: -kv[1])[:7]:
        y += 30
        cv2.putText(stats, f"{k}: {n}", (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
    bottom = np.hstack([legend, tiles.get("CAM_BACK", blank), stats])
    return np.vstack([top, bottom])

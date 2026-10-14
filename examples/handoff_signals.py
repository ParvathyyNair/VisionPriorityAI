"""Handoff test: prove the tracker output contains what the risk layer needs.

This is NOT the risk model (that is Parvati's module). It only computes the raw
motion signals from real tracked frames, to show that:
  * A (approach rate)  can be derived from area_history  (bbox area growth)
  * D (direction)      can be derived from centroid_history (motion vector)
  * P (persistence)    can be derived from track_age / hits
  * Z (zone)           can be derived from centroid + image_size
and that the values look sensible (objects getting bigger have positive growth).

  python examples/handoff_signals.py results/nuscenes/scene00/tracks.jsonl --frame 120
"""
from __future__ import annotations

import argparse
import json

import numpy as np


def area_growth_per_s(area_history, window_s=1.0):
    """Slope of log(area) per second over the last `window_s` seconds.
    > 0 means the box is growing (object getting closer), ~0 static, < 0 receding.
    Using log makes it scale-free: +0.69/s = area doubles every second."""
    h = np.array(area_history, float)
    if len(h) < 3:
        return 0.0
    h = h[h[:, 0] >= h[-1, 0] - window_s]
    if len(h) < 3 or np.ptp(h[:, 0]) == 0:
        return 0.0
    return float(np.polyfit(h[:, 0], np.log(np.maximum(h[:, 1], 1.0)), 1)[0])


def centroid_velocity(centroid_history, window_s=1.0):
    """(vx, vy) in pixels/second over the last window."""
    h = np.array(centroid_history, float)
    if len(h) < 3:
        return 0.0, 0.0
    h = h[h[:, 0] >= h[-1, 0] - window_s]
    if len(h) < 3 or np.ptp(h[:, 0]) == 0:
        return 0.0, 0.0
    return float(np.polyfit(h[:, 0], h[:, 1], 1)[0]), float(np.polyfit(h[:, 0], h[:, 2], 1)[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl")
    ap.add_argument("--frame", type=int, default=None, help="frame to inspect (default: busiest frame)")
    ap.add_argument("--save-sample", default=None, help="write N frames around --frame as a small JSON for unit tests")
    a = ap.parse_args()
    frames = [json.loads(l) for l in open(a.jsonl)]
    fi = a.frame if a.frame is not None else max(range(len(frames)), key=lambda i: len(frames[i]["tracks"]))
    fr = frames[fi]
    print(f"frame {fr['frame_idx']}  t={fr['timestamp']}s  active tracks={len(fr['tracks'])}\n")
    rows = []
    for t in fr["tracks"]:
        g = area_growth_per_s(t["area_history"])
        vx, vy = centroid_velocity(t["centroid_history"])
        w, h = t["image_size"]
        rows.append((g, t["track_label"], t["class"], t["confidence"], t["track_age"], t["hits"],
                     t["centroid"][0] / w, t["centroid"][1] / h, t["bbox_area_norm"], vx, vy))
    rows.sort(key=lambda r: -r[0])
    print(f"{'label':8} {'class':11} {'conf':>5} {'age':>4} {'hits':>4} {'cx/W':>5} {'cy/H':>5} "
          f"{'area%':>6} {'growth/s':>8} {'vx px/s':>8} {'vy px/s':>8}")
    for g, lab, c, conf, age, hits, nx, ny, an, vx, vy in rows:
        print(f"{lab:8} {c:11} {conf:5.2f} {age:4d} {hits:4d} {nx:5.2f} {ny:5.2f} {an*100:6.2f} {g:8.3f} {vx:8.1f} {vy:8.1f}")
    if a.save_sample:
        lo, hi = max(0, fi - 12), min(len(frames), fi + 1)
        json.dump(frames[lo:hi], open(a.save_sample, "w"))
        print(f"\nsaved frames {lo}..{hi-1} to {a.save_sample}")


if __name__ == "__main__":
    main()

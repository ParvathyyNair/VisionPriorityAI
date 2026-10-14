"""Render ByteTrack IDs on a MOT17 video (from the cached YOLO11s detections) for visual spot-checks.

  python scripts/render_mot17_tracks.py --video path/MOT17-09-FRCNN.mp4 --cache results/mot17_cache \
      --out results/mot17/MOT17-09_bytetrack.mp4

Each ID keeps one colour, so an ID switch is visible as a colour change on the same person.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_mot17 import run_bytetrack  # noqa: E402


def color(i):
    rng = np.random.default_rng(int(i) * 7919)
    return tuple(int(c) for c in rng.integers(60, 255, 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--cache", default="results/mot17_cache")
    ap.add_argument("--out", required=True)
    ap.add_argument("--width", type=int, default=960)
    a = ap.parse_args()
    seq = Path(a.video).stem
    c = dict(np.load(Path(a.cache) / f"yolo11s_{seq}_s1.npz", allow_pickle=True))
    tracks = run_bytetrack(c)
    cap = cv2.VideoCapture(a.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    tmp = Path(a.out).with_suffix(".tmp.mp4")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    w = None
    i = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        boxes, ids = tracks.get(i, (np.zeros((0, 4)), np.zeros(0, int)))
        for b, t in zip(boxes, ids):
            x1, y1, x2, y2 = map(int, b)
            cv2.rectangle(f, (x1, y1), (x2, y2), color(t), 3)
            cv2.putText(f, f"ID {t}", (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color(t), 2, cv2.LINE_AA)
        cv2.rectangle(f, (0, 0), (f.shape[1], 44), (20, 20, 20), -1)
        cv2.putText(f, f"{seq}  frame {i}  YOLO11s + ByteTrack  active IDs: {len(ids)}", (12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        s = a.width / f.shape[1]
        f = cv2.resize(f, (a.width, int(f.shape[0] * s)))
        if w is None:
            w = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (f.shape[1], f.shape[0]))
        w.write(f)
        i += 1
    w.release()
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-crf", "27", "-preset", "veryfast", a.out], check=True)
    tmp.unlink()
    print("saved", a.out)


if __name__ == "__main__":
    main()

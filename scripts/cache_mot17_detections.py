"""Run a YOLO11 model over MOT17 videos once and cache the person detections.

Caching lets the tracking evaluation try several ByteTrack settings without
re-running the detector. Output: one .npz per (model, sequence) holding an
object array `dets` (per frame, (N,6) = x1,y1,x2,y2,conf,cls) plus timing.

Usage:
  python scripts/cache_mot17_detections.py --videos <mot17_videos dir> \
      --models yolo11s.pt --stride 1 --out results/mot17_cache
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vp_perception.config import DetectorConfig  # noqa: E402
from vp_perception.detector import YoloDetector  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True)
    ap.add_argument("--models", nargs="+", default=["yolo11s.pt"])
    ap.add_argument("--stride", type=int, default=1, help="process every k-th frame (1 = all)")
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--seqs", nargs="*", default=None)
    ap.add_argument("--out", default="results/mot17_cache")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    vids = sorted(Path(a.videos).glob("MOT17-*.mp4"))
    if a.seqs:
        vids = [v for v in vids if any(s in v.name for s in a.seqs)]
    for m in a.models:
        det = YoloDetector(DetectorConfig(weights=m, conf=a.conf, classes=(0,)))
        for v in vids:
            tag = f"{Path(m).stem}_{v.stem}_s{a.stride}"
            dst = out / f"{tag}.npz"
            if dst.exists():
                print("skip", dst)
                continue
            cap = cv2.VideoCapture(str(v))
            frames_dets, idxs, times = [], [], []
            i = 0
            while True:
                ok, f = cap.read()
                if not ok:
                    break
                if i % a.stride == 0:
                    d = det.detect(f)
                    times.append(det.last_ms)
                    frames_dets.append(np.concatenate([d.xyxy, d.conf[:, None], d.cls[:, None]], 1))
                    idxs.append(i)
                i += 1
            arr = np.empty(len(frames_dets), dtype=object)
            arr[:] = frames_dets
            np.savez_compressed(dst, dets=arr, frame_idx=np.array(idxs), ms=np.array(times), n_frames=i)
            print(f"{tag}: {len(idxs)} frames, {np.median(times[1:]):.1f} ms/frame median", flush=True)


if __name__ == "__main__":
    main()

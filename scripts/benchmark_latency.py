"""Measure YOLO11 n / s / m latency on YOUR hardware, on real nuScenes frames.

Run this on the laptop/GPU you will demo on — the cloud numbers in the report
were measured on a 2-core CPU with no GPU.

  python scripts/benchmark_latency.py --videos <nuscenes_videos dir> --scene scene00 --device 0
  python scripts/benchmark_latency.py --videos <nuscenes_videos dir> --scene scene00 --device cpu

Reports, per model: ms per single view, ms per 4-view batch (what the pipeline
actually does each frame), effective FPS, and whether the 4-view batch meets
the PDS-32 <= 100 ms processing target. Writes results/latency_<device>.json.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vp_perception import MultiViewVideoSource, find_scene_videos  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True)
    ap.add_argument("--scene", default="scene00")
    ap.add_argument("--models", nargs="+", default=["yolo11n.pt", "yolo11s.pt", "yolo11m.pt"])
    ap.add_argument("--device", default=None)
    ap.add_argument("--frames", type=int, default=30)
    ap.add_argument("--half", action="store_true", help="FP16 (GPU only)")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    import torch
    from ultralytics import YOLO

    src = MultiViewVideoSource(find_scene_videos(a.videos, a.scene), max_frames=a.frames)
    batches = [list(b.frames.values()) for b in src]
    dev = a.device if a.device is not None else ("0" if torch.cuda.is_available() else "cpu")
    hw = torch.cuda.get_device_name(0) if dev != "cpu" and torch.cuda.is_available() else platform.processor() or "CPU"
    res = {"device": dev, "hardware": hw, "torch": torch.__version__, "imgsz": 640, "frames": len(batches), "models": {}}
    for m in a.models:
        y = YOLO(m)
        kw = dict(imgsz=640, conf=0.1, verbose=False, device=dev)
        if a.half:
            kw["half"] = True
        for _ in range(3):
            y.predict(batches[0], **kw)  # warm-up
        single, batch = [], []
        for fr in batches:
            t = time.perf_counter(); y.predict(fr[0], **kw); single.append((time.perf_counter() - t) * 1000)
            t = time.perf_counter(); y.predict(fr, **kw); batch.append((time.perf_counter() - t) * 1000)
        r = {"ms_single_view_median": round(float(np.median(single)), 1),
             "ms_4view_batch_median": round(float(np.median(batch)), 1),
             "ms_4view_batch_p95": round(float(np.percentile(batch, 95)), 1),
             "fps_4view": round(1000 / float(np.median(batch)), 1),
             "meets_100ms_target": bool(np.median(batch) <= 100)}
        res["models"][Path(m).stem] = r
        print(Path(m).stem, r, flush=True)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    p = Path(a.out) / f"latency_{dev}.json"
    p.write_text(json.dumps(res, indent=2))
    print("saved", p)


if __name__ == "__main__":
    main()

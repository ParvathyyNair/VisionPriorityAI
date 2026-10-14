"""Run Detect + Track on nuScenes scene videos (4 synchronized views).

Outputs per scene in <out>/<scene>/:
  tracks.jsonl          one JSON line per synchronized frame: {frame_idx, timestamp, timing_ms, tracks:[...]}
  tracks_flat.csv       one row per track per frame (contract fields + key extras) for quick inspection
  meta.json             config, sync mode, frame counts, latency summary, track statistics
  surround_tracks.mp4   annotated surround mosaic (H.264)

Usage:
  python scripts/run_nuscenes.py --videos <nuscenes_videos dir> --scenes scene00 scene02 --out results/nuscenes
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vp_perception import MultiViewVideoSource, find_scene_videos, validate_record, CONTRACT_FIELDS  # noqa: E402
from vp_perception.config import PerceptionConfig, DetectorConfig  # noqa: E402
from vp_perception.pipeline import PerceptionPipeline  # noqa: E402
from vp_perception.viz import surround_mosaic  # noqa: E402

FLAT_EXTRA = ["frame_idx", "timestamp", "track_label", "class_group", "bbox_area_norm", "track_age_s", "hits"]


def run_scene(pipe, videos_dir, scene, out_root, max_frames=None, save_video=True, sync="auto"):
    out = Path(out_root) / scene
    out.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        src = MultiViewVideoSource(find_scene_videos(videos_dir, scene, pipe.views), fps=pipe.cfg.fps,
                                   sync=sync, max_frames=max_frames)
        sync_warnings = [str(x.message) for x in w]
    pipe.reset()
    tmp_video = out / "_tmp.mp4"
    writer = None
    timings, per_frame_counts, errors = [], [], []
    track_life = defaultdict(lambda: {"frames": 0, "class": None, "first": None, "last": None})
    with open(out / "tracks.jsonl", "w") as fj, open(out / "tracks_flat.csv", "w", newline="") as fc:
        cw = csv.writer(fc)
        cw.writerow(list(CONTRACT_FIELDS) + FLAT_EXTRA)
        for b in src:
            r = pipe.process(b)
            fj.write(json.dumps(r.to_dict()) + "\n")
            timings.append(r.timing_ms)
            per_frame_counts.append(len(r.tracks))
            for t in r.tracks:
                e = validate_record(t)
                if e:
                    errors.append((t["frame_idx"], t["track_label"], e))
                cw.writerow([t["view"], t["track_id"], t["class"], json.dumps(t["bbox"]), t["confidence"],
                             json.dumps(t["centroid"]), t["bbox_area"], t["track_age"]] + [t[k] for k in FLAT_EXTRA])
                L = track_life[(t["view"], t["track_id"])]
                L["frames"] += 1
                L["class"] = t["class"]
                L["first"] = t["first_frame_idx"] if L["first"] is None else L["first"]
                L["last"] = t["frame_idx"]
            if save_video:
                hdr = f"{scene}  frame {b.frame_idx}  t={b.timestamp:.2f}s\nsync: {b.sync_mode}\n" \
                      f"detect {r.timing_ms['detect']:.0f} ms (4 views)"
                m = surround_mosaic(b.frames, r.tracks, hdr)
                if writer is None:
                    writer = cv2.VideoWriter(str(tmp_video), cv2.VideoWriter_fourcc(*"mp4v"), pipe.cfg.fps,
                                             (m.shape[1], m.shape[0]))
                writer.write(m)
            if b.frame_idx % 25 == 0:
                print(f"  {scene} frame {b.frame_idx}/{len(src)}  tracks={len(r.tracks)}  "
                      f"{r.timing_ms['total']:.0f} ms", flush=True)
    if writer is not None:
        writer.release()
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp_video), "-c:v", "libx264",
                        "-pix_fmt", "yuv420p", "-crf", "26", "-preset", "veryfast", str(out / "surround_tracks.mp4")],
                       check=True)
        tmp_video.unlink()

    det = np.array([t["detect"] for t in timings[1:]] or [0])  # skip warm-up frame
    tot = np.array([t["total"] for t in timings[1:]] or [0])
    lens = np.array([L["frames"] for L in track_life.values()] or [0])
    by_view = defaultdict(int)
    by_class = defaultdict(int)
    for (v, _), L in track_life.items():
        by_view[v] += 1
        by_class[L["class"]] += 1
    meta = {
        "scene": scene,
        "videos": {v: str(p.name) for v, p in src.videos.items()},
        "frame_counts_per_view": src.counts,
        "sync_mode": src.sync,
        "sync_warnings": sync_warnings,
        "n_batches": len(timings),
        "config": pipe.cfg.to_dict(),
        "latency_ms": {
            "device": str(pipe.detector.model.device),
            "detect_4views_median": round(float(np.median(det)), 1),
            "detect_4views_p95": round(float(np.percentile(det, 95)), 1),
            "detect_per_view_median": round(float(np.median(det)) / len(pipe.views), 1),
            "total_median": round(float(np.median(tot)), 1),
            "track_median": round(float(np.median([t["track"] for t in timings])), 2),
        },
        "tracks": {
            "unique_ids": len(track_life),
            "by_view": dict(by_view),
            "by_class": dict(sorted(by_class.items(), key=lambda kv: -kv[1])),
            "mean_active_per_frame": round(float(np.mean(per_frame_counts)), 2),
            "median_track_length_frames": float(np.median(lens)),
            "tracks_longer_than_1s": int((lens >= pipe.cfg.fps).sum()),
            "single_frame_tracks": int((lens == 1).sum()),
        },
        "contract_validation_errors": len(errors),
        "contract_error_examples": errors[:5],
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({k: meta[k] for k in ("scene", "n_batches", "latency_ms", "tracks", "contract_validation_errors")}, indent=1))
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True, help="folder with sceneXX_CAM_*.mp4")
    ap.add_argument("--scenes", nargs="+", required=True)
    ap.add_argument("--out", default="results/nuscenes")
    ap.add_argument("--weights", default="yolo11s.pt")
    ap.add_argument("--device", default=None, help="e.g. 0 for first GPU, cpu")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--sync", default="auto", choices=["auto", "timestamps", "proportional", "index"])
    a = ap.parse_args()
    cfg = PerceptionConfig(detector=DetectorConfig(weights=a.weights, device=a.device))
    pipe = PerceptionPipeline(cfg)
    for s in a.scenes:
        run_scene(pipe, a.videos, s, a.out, a.max_frames, not a.no_video, a.sync)


if __name__ == "__main__":
    main()

"""Heavy-vehicle extension: add the two rear-side cameras (CAM_BACK_LEFT, CAM_BACK_RIGHT).

For each scene this
  1. runs YOLO11s + one ByteTrack per view on the two extra cameras, using the same
     proportional frame alignment as the original four views (master clock = CAM_FRONT),
  2. writes results/nuscenes/<scene>/tracks_rear_sides.jsonl (same Output Contract, one line per frame),
  3. renders results/nuscenes/<scene>/surround6_tracks.mp4: a 3x2 six-camera mosaic
        [ FRONT-LEFT | FRONT | FRONT-RIGHT ]
        [ REAR-LEFT  | REAR  | REAR-RIGHT  ]
     with the tracks of all six cameras drawn on it.

The original four-camera files (tracks.jsonl, surround_tracks.mp4) are not modified.

  python scripts/extend_heavy_views.py --videos <folder with sceneXX_CAM_*.mp4> --scenes scene00 scene01
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vp_perception import MultiViewVideoSource  # noqa: E402
from vp_perception.config import PerceptionConfig, DetectorConfig  # noqa: E402
from vp_perception.pipeline import PerceptionPipeline  # noqa: E402
from vp_perception.viz import draw_tracks  # noqa: E402

ALL6 = ("CAM_FRONT", "CAM_BACK", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK_LEFT", "CAM_BACK_RIGHT")
NEW = ["CAM_BACK_LEFT", "CAM_BACK_RIGHT"]
GRID = [["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT"], ["CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"]]
NAMES = {"CAM_FRONT_LEFT": "FRONT-LEFT", "CAM_FRONT": "FRONT", "CAM_FRONT_RIGHT": "FRONT-RIGHT",
         "CAM_BACK_LEFT": "REAR-LEFT", "CAM_BACK": "REAR", "CAM_BACK_RIGHT": "REAR-RIGHT"}
TILE = (533, 300)


def tile(img, title, tracks):
    t = cv2.resize(draw_tracks(img, tracks), TILE, interpolation=cv2.INTER_AREA)
    cv2.rectangle(t, (0, 0), (TILE[0], 26), (25, 25, 25), -1)
    cv2.putText(t, f"{title}  |  {len(tracks)} tracks", (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return t


def run_scene(pipe, videos_dir: Path, scene: str):
    out = ROOT / "results" / "nuscenes" / scene
    base = [json.loads(l) for l in open(out / "tracks.jsonl", encoding="utf-8") if l.strip()]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        src = MultiViewVideoSource({v: videos_dir / f"{scene}_{v}.mp4" for v in ALL6}, fps=pipe.cfg.fps, sync="auto")
    assert len(src) == len(base), f"{scene}: {len(src)} synced frames vs {len(base)} in tracks.jsonl"
    pipe.reset()
    tmp = out / "_tmp6.mp4"
    writer = None
    n_rec = 0
    with open(out / "tracks_rear_sides.jsonl", "w", encoding="utf-8") as fj:
        for b, fr in zip(src, base):
            r = pipe.process(b)
            fj.write(json.dumps(r.to_dict()) + "\n")
            n_rec += len(r.tracks)
            by_view = {}
            for t in fr["tracks"] + r.tracks:
                by_view.setdefault(t["view"], []).append(t)
            mosaic = np.vstack([np.hstack([tile(b.frames[v], NAMES[v], by_view.get(v, [])) for v in row]) for row in GRID])
            if writer is None:
                writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), pipe.cfg.fps, (mosaic.shape[1], mosaic.shape[0]))
            writer.write(mosaic)
            if b.frame_idx % 50 == 0:
                print(f"  {scene} frame {b.frame_idx}/{len(src)}  rear-side tracks={len(r.tracks)}", flush=True)
    writer.release()
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-profile:v", "high",
                    "-crf", "26", "-preset", "veryfast", "-movflags", "+faststart", str(out / "surround6_tracks.mp4")], check=True)
    tmp.unlink()
    print(f"{scene}: {len(base)} frames, {n_rec} rear-side track records, counts {src.counts}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True)
    ap.add_argument("--scenes", nargs="+", required=True)
    ap.add_argument("--weights", default="yolo11s.pt")
    a = ap.parse_args()
    pipe = PerceptionPipeline(PerceptionConfig(detector=DetectorConfig(weights=a.weights)), views=NEW)
    for sc in a.scenes:
        run_scene(pipe, Path(a.videos), sc)


if __name__ == "__main__":
    main()

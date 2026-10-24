# VisionPriority AI — Detection & Tracking module (Review 2)

**Owner:** Harini (23BCE2170)  **Pipeline stages:** 1 INPUT → 2 PREPROCESS → 3 DETECT → 4 TRACK
**Consumes:** Aishwarya's nuScenes per-camera videos (and MOT17 for validation)
**Produces:** Output Contract records for Parvati's ANALYZE → RISK → PRIORITIZE → EXPLAIN stages

```
4 synchronized views ──► YOLO11s (COCO, 640) ──► ByteTrack ×4 (one per view) ──► Output Contract records
 (SyncedFrameBatch)      one batched pass/frame    two-stage association             view • track_id • class • bbox •
                                                   + low-score recovery              confidence • centroid • bbox_area •
                                                                                     track_age  (+ histories for A / D)
```

## Folder layout

| Path | What it is |
|---|---|
| `vp_perception/config.py` | Every tunable value: views, kept COCO classes, YOLO and ByteTrack settings |
| `vp_perception/frame_source.py` | `SyncedFrameBatch` (the interface with Aishwarya's loader) + `MultiViewVideoSource` for her .mp4 files |
| `vp_perception/detector.py` | YOLO11s wrapper — all views of one instant in one batched forward pass |
| `vp_perception/tracker.py` | `ViewTracker` — ByteTrack per view + the per-track history the risk layer needs |
| `vp_perception/contract.py` | **The Output Contract** — field names, `TrackRecord`, `validate_record()` |
| `vp_perception/pipeline.py` | `PerceptionPipeline.process(batch)` → `FrameResult` (what Parvati calls) |
| `vp_perception/viz.py` | Box overlays + surround mosaic video |
| `scripts/run_nuscenes.py` | Run on nuScenes scenes → `tracks.jsonl`, `tracks_flat.csv`, `meta.json`, `surround_tracks.mp4` |
| `scripts/cache_mot17_detections.py` + `scripts/eval_mot17.py` | Detector (n/s/m) and tracker validation against MOT17 ground truth |
| `scripts/benchmark_latency.py` | Latency of YOLO11 n/s/m on **your** machine (run before the demo) |
| `scripts/check_contract.py` | Validates any `tracks.jsonl` against the contract |
| `examples/handoff_signals.py` | Shows A / D / P / Z raw signals computed from real tracker output |
| `tests/test_perception.py` | Unit tests (no weights or videos needed) |
| `results/` | Outputs produced for Review 2 (see `docs/HANDOFF.md`) |

## Setup

```bash
# 1. PyTorch: GPU build from https://pytorch.org if you have an NVIDIA GPU, otherwise:
pip install torch torchvision
# 2. The rest
pip install -r requirements.txt
# 3. Tests (should print 5 x ok)
python tests/test_perception.py
```
YOLO11 weights (`yolo11s.pt` etc.) download automatically on first use.

## Run

```bash
# nuScenes, 4 views, all frames, with annotated video
python scripts/run_nuscenes.py --videos path/to/nuscenes_videos --scenes scene00 scene01 --out results/nuscenes
# add --device 0 to force the GPU, --max-frames 50 for a quick test, --no-video to skip the mosaic

# check the output against the contract
python scripts/check_contract.py results/nuscenes/scene00/tracks.jsonl

# latency on your hardware (do this on the demo laptop)
python scripts/benchmark_latency.py --videos path/to/nuscenes_videos --scene scene00

# MOT17 validation (slow on CPU: caches detections first)
python scripts/cache_mot17_detections.py --videos path/to/mot17_videos --models yolo11s.pt --stride 1
python scripts/cache_mot17_detections.py --videos path/to/mot17_videos --models yolo11n.pt yolo11m.pt --stride 3
python scripts/eval_mot17.py --gt path/to/mot17_gt
```

## Using it from Python (Parvati's integration)

```python
from vp_perception import MultiViewVideoSource, find_scene_videos
from vp_perception.pipeline import PerceptionPipeline

pipe = PerceptionPipeline()                                    # YOLO11s + 4 ByteTrackers
src = MultiViewVideoSource(find_scene_videos("nuscenes_videos", "scene00"))
for batch in src:                                              # one synchronized instant
    result = pipe.process(batch)
    for rec in result.tracks:                                  # plain dicts, contract field names
        risk_layer.update(rec)                                 # ← Parvati's module
```

Full field definitions and how each risk factor maps onto them: **`docs/HANDOFF.md`**.

## Parvati's Review 2 risk layer + dashboard

```bash
pip install -r requirements.txt
python -m pytest tests/test_risk_layer.py tests/test_advisory.py -q
streamlit run app.py
```

The dashboard uses the **precomputed** `results/nuscenes/sceneXX/tracks.jsonl` and `surround_tracks.mp4`, so it does not rerun YOLO.

**Live console.** Press play on the annotated video and the page reads the video clock (one tracked frame per video frame at 12 fps).
On every frame it updates:
- an advisory banner with a short driver action ("BRAKE READY", "CHECK LEFT", ...) and the object behind it,
- a top-down threat map for the four camera directions,
- the ranked priority feed and a live factor breakdown (C, Z, A, D, P) for the strongest alert,
- an escalation log of critical alerts raised so far (click an entry to jump to it),
- a risk timeline for the whole clip (click or drag to seek).

A generated two-tone sound plays when a new critical alert is raised during playback (toggle it with the Sound checkbox; browsers allow audio once you have pressed play).
The banner holds the strongest alert for about 0.4 s so it does not flicker between frames.

**Light and dark mode.** Use the switch in the sidebar. The choice is kept in the URL (`?theme=light`) and switching it does not restart the video.

**Frame inspector** keeps the earlier slider-based view: annotated frame, ranked alerts, factor breakdown, full track table and CSV download.

Files: `app.py` (page), `vp_ui/theme.py` (light/dark tokens), `vp_ui/console.html` (live console), `vp_ui/payload.py` (scores every frame once),
`vp_ui/advisory.py` (driver wording), `.streamlit/config.toml` (static serving for the video).
`risk_layer.py` is unchanged.

Risk logic is in `risk_layer.py`. The score is a transparent heuristic, not a calibrated probability or collision prediction. Z uses image position as a
proximity proxy; D uses view-specific image-plane direction heuristics. The driver advice is rule-based text on top of those scores and has not been validated.
State these assumptions in the demo.

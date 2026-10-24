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



## Heavy-vehicle mode: more cameras for bigger blind spots

Buses, coaches and trucks have much larger blind spots than cars: alongside the long body, directly
below the windscreen, and directly behind. Pick the vehicle in the sidebar, either by **type** or by
**number of wheels**. The app then changes the camera set, the risk weighting and the driver advice.

| Vehicle | Wheels | Length | Cameras | What changes |
|---|---|---|---|---|
| Car / SUV | 4 | 4.5 m | 4 | Original configuration (scores unchanged) |
| Van / mini-truck | 4-6 | 6.5 m | 6 | + rear-left / rear-right cameras, blind-spot gains ×1.05-1.20 |
| City bus | 6 | 12 m | 6 | gains ×1.10-1.30, front blind zone under the windscreen |
| Coach bus | 6-8 | 13.5 m | 6 | gains ×1.10-1.35 |
| Heavy truck / trailer | 10-18 | 16.5 m | 6 | gains ×1.15-1.40, largest on the left (near side in India) |

* **Real footage from 6 cameras.** nuScenes records six cameras. Heavy profiles add `CAM_BACK_LEFT`
  and `CAM_BACK_RIGHT`, processed with the same YOLO11s + ByteTrack pipeline
  (`scripts/extend_heavy_views.py` → `results/nuscenes/<scene>/tracks_rear_sides.jsonl` and
  `surround6_tracks.mp4`, a 3×2 six-camera mosaic).
* **Blind-spot weighting** (`vehicle_profiles.py`): the zone factor **Z** of each camera is multiplied by
  the profile's gain, capped at 1.0. Side-camera objects that reach Z ≥ 0.6 are labelled `blind-spot`.
  The formula weights and the priority thresholds are unchanged, and the car profile gives exactly the
  original scores.
* **Heavy-vehicle alerts** (`vp_ui/advisory.py`): `BLIND SPOT LEFT/RIGHT`, `WATCH LEFT/RIGHT FLANK`,
  `DO NOT MOVE OFF` (vulnerable road user below the windscreen), `NO LEFT TURN` (trailer / long-body
  cut-in), `DO NOT REVERSE`, `CHECK REAR`, `BRAKE EARLY` (longer stopping distance).
* **Live console:** a six-sector threat map with a bus, truck or van silhouette, the six-camera
  video, and the new alerts in the banner, priority feed and escalation log.
* Works **online (same Streamlit link) and offline**. `python scripts/offline_check.py` tests every
  vehicle type with the network blocked.

The blind-spot gains are prototype design values, informed by the larger mirror blind spots of heavy
vehicles. They have not been calibrated against measured blind-spot maps of specific vehicle models.
The footage comes from nuScenes' six-camera car rig and is used to emulate a heavy-vehicle camera layout.

## Offline mode: works with no internet connection

Everything VisionPriority AI needs runs on the local computer, so a driver in an area with no
mobile signal gets the same live console, advisories and threat map as one with full connectivity.

| Component | How it works offline |
|---|---|
| App server | Streamlit runs locally at `http://localhost:8501`; the browser talks only to this computer |
| Scene videos | served from this computer (`static/`, `results/nuscenes/*/surround_tracks.mp4`) |
| Risk scores, advisories, escalation log | computed locally from `results/nuscenes/*/tracks.jsonl` by `risk_layer.py` + `vp_ui/` |
| Fonts (Chakra Petch, Manrope) | bundled in `assets/fonts/` and embedded in the page (no Google Fonts request) |
| Detector weights (YOLO11s / YOLO11n) | bundled in `models/`; never downloaded at runtime (`YOLO_OFFLINE=true`) |
| Alert sound | generated in the browser (Web Audio), no network |
| Usage telemetry | disabled in `.streamlit/config.toml` |

**Install once with internet:** `pip install -r requirements.txt`

**Install on a laptop with no internet:**
1. On any computer with internet (same OS and Python version), run `build_offline_installer.bat`
   (Windows) or `./build_offline_installer.sh`. This downloads every dependency into `wheelhouse/`.
2. Copy the whole project folder, including `wheelhouse/`, to the offline laptop (USB is fine).
3. Run `install_offline.bat` or `./install_offline.sh`.

**Run:** double-click `run_offline.bat` (Windows) or run `./run_offline.sh`. It runs a quick offline
self-check, then opens the dashboard in the browser.

**Prove it at any time:** `python scripts/offline_check.py` blocks all network access and then tests
every scene in dark and light mode with every alert filter. It also checks that the UI contains no
internet links, that advisories are generated for every frame, that YOLO11s loads from `models/`,
and that **0 network connections** are attempted.

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

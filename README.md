# VisionPriority AI

### 360° Situational Awareness & Driver Assistance

**Goal:** Build a software-based computer-vision prototype that
processes four surround-view camera feeds, detects and tracks road
users, assigns explainable threat scores, and prioritizes alerts.

**Review 2 roadmap:** Dataset pipeline → Detection & tracking → Risk
scoring → Dashboard → Measured validation → Demo.

## End-to-end architecture

1.  Input: four nuScenes views --- `CAM_FRONT`, `CAM_BACK`,
    `CAM_FRONT_LEFT`, `CAM_FRONT_RIGHT`.
2.  Preprocess: standardize native 1600×900 frames for 640×640 detector
    input.
3.  Detect: COCO-pretrained YOLO11s.
4.  Track: ByteTrack, one tracker per view; cross-camera
    re-identification is out of scope.
5.  Analyze: derive spatial and temporal features.
6.  Risk: calculate a custom rule-based threat score.
7.  Prioritize: rank active tracks by score.
8.  Explain: generate a plain-language reason for each alert.

## Team work division

### Aishwarya --- Dataset Pipeline

**Primary responsibility:** Prepare synchronized multi-view data and
provide a stable input interface.

-   **nuScenes primary dataset**
    -   Move from nuScenes-mini development data to the relevant
        `trainval` portion for evaluation, as planned in the deck.
    -   Set up `nuscenes-devkit`; verify the four selected camera views
        and calibration.
    -   Synchronize camera frames by nuScenes `sample` timestamp so the
        views represent the same instant.
    -   Build an FFmpeg stitching script for four-view video across
        required scenes.
    -   Document the resolution handoff: native 1600×900 → OpenCV
        preprocessing → 640×640 detector input.
-   **Supporting datasets**
    -   WoodScape: retain documentation as supporting/future work; no
        extra pipeline integration unless agreed.
    -   BDD100K: finalize documentation for the 100-image detection
        sample and collected raw video clips.
    -   MOT17: produce a stitched 600-frame tracking-validation MP4 for
        ByteTrack testing.
-   **Data loader and handoff**
    -   Implement the Python loader supplying synchronized four-view
        frames to the detector.
    -   Include `view` and `timestamp`; document folder structure and
        paths.
    -   Agree with Harini on file paths versus in-memory arrays,
        timestamp format, and loader return structure.

**Deliverables:** Dataset loader, synchronized/stiched four-view input,
supporting-dataset documentation, MOT17 validation video, and
input-interface documentation.

### Harini --- Detection & Tracking Pipeline

**Primary responsibility:** Detect road users in each view and maintain
track identities over time.

-   Integrate Ultralytics YOLO11s with COCO-pretrained weights and
    640×640 input.
-   Run detection independently on each synchronized camera view.
-   Output detection fields `class`, `bbox`, and `confidence`.
-   Benchmark YOLO11n/s/m on the team's actual dataset and hardware;
    published figures are references, not project measurements.
-   Implement ByteTrack independently per view.
-   Check ID continuity through occlusion and the effect of
    low-confidence-box association.
-   Extend tracks with `track_id`, `centroid`, `bbox_area`, and
    `track_age`.
-   Validate with the agreed input and tracking-validation video;
    document limitations.
-   Confirm field names and coordinate conventions match the shared
    Output Contract before handing off to Parvati.

**Deliverables:** Detection/tracking pipeline, per-track JSONL and
annotated video, measured benchmark results, and contract validation.

### Parvati --- Integration, Risk Logic, Dashboard & Delivery

**Primary responsibility:** Connect the modules, implement explainable
risk reasoning, and prepare the Review 2 demo.

**1. Risk and explanation module** - Implement five factors: class
vulnerability (**C**), zone criticality (**Z**), approach rate (**A**),
direction alignment (**D**), and persistence (**P**). - Apply the
formula exactly:

`Risk = 0.25C + 0.20Z + 0.30A + 0.15D + 0.10P`

-   Priority bands:
    -   **LOW:** 0.00--0.34
    -   **MEDIUM:** 0.35--0.64
    -   **HIGH/CRITICAL:** 0.65--1.00
-   Generate template-based plain-language explanations from the same
    factors used for scoring, e.g. "Person approaching from left-side
    critical zone."
-   Verify actual tracker field names and coordinate conventions.
    Approach rate needs bounding-box area history; persistence uses
    track age/history.

**2. End-to-end integration** - Connect Input → Preprocess → Detect →
Track → Analyze → Risk → Prioritize → Explain. - Enforce the Output
Contract as the real interface between stages. - Rank active tracks by
score and surface the highest-priority hazard first. - Own the shared
GitHub repository structure and integration workflow.

**3. Streamlit dashboard** - Build the ranked surround-view alert feed
with score, view/location, track ID, priority, and reason. - Build a
simplified surround-awareness map. - Show track and risk-factor details
where available. - Clearly label whether the demo uses precomputed
results or live inference.

**4. Validation and delivery** - Measure processing latency against the
target of ≤100 ms. - Measure end-to-end time to audio-visual alert
against the target of within 1 second. - Record hardware, dataset, input
resolution, timing method, and observed results. Do not present targets
or published benchmarks as measured outcomes. - Rehearse the full path
from stitched nuScenes input to prioritized alerts. - Prepare Review 2
slides and demo.

**Deliverables:** Risk/explanation module, integration layer, Streamlit
dashboard, measured validation report, GitHub integration, slides, and
demo rehearsal.

## Critical handoffs

### Aishwarya → Harini: synchronized input

Agree on: - File paths versus in-memory arrays. - Timestamp format and
how views are grouped. - Camera names, frame dimensions, and loader
return structure. - Final folder paths.

**Acceptance check:** One time step provides all four expected views
with consistent timestamps and documented image dimensions.

### Harini → Parvati: tracked-object contract

Core fields: - `view` - `track_id` - `class` - `bbox` - `confidence` -
`centroid` - `bbox_area` - `track_age`

Confirm exact names, coordinate conventions, and any history fields used
by the risk module.

**Acceptance check:** Pass several real tracked frames through risk
scoring and verify that all five factors, the score, priority, and
explanation are generated without missing-field errors.

## Shared Output Contract

`view • track_id • class • bbox • confidence • centroid • bbox_area • track_age`

The risk layer derives its five factors from the tracked object and
available history. The final alert contains the score, priority,
view/location, track ID, and plain-language explanation.

## Review 2 checklist

### Aishwarya

-   [ ] Finalize nuScenes evaluation subset and calibration checks.
-   [ ] Verify synchronization across the four selected views.
-   [ ] Finalize the stitching script and Python loader.
-   [ ] Document WoodScape, BDD100K, and MOT17 assets.
-   [ ] Publish the agreed input interface and folder structure.

### Harini

-   [ ] Verify YOLO11s on synchronized four-view input.
-   [ ] Benchmark YOLO11n/s/m on the actual setup.
-   [ ] Validate ByteTrack ID continuity and occlusion behavior.
-   [ ] Export and validate track JSONL and annotated video.
-   [ ] Confirm the Output Contract with Parvati.

### Parvati

-   [ ] Validate risk factors and score on real tracker outputs.
-   [ ] Test priority thresholds, ranking, and explanations.
-   [ ] Connect input, perception, risk, and dashboard stages.
-   [ ] Test the dashboard with agreed demo data.
-   [ ] Measure processing and alert latency.
-   [ ] Prepare Review 2 slides and rehearse the demo.

## Performance targets and scope

The source problem statement specifies **≤100 ms processing latency**
and **audio-visual alerts within 1 second**. These are targets to
measure, not results to claim in advance.

-   Tracking is per camera; cross-camera re-identification is out of
    scope.
-   Image-coordinate proximity is a visual proxy, not calibrated metric
    distance.
-   Risk scoring is custom rule-based logic, not an additional learned
    model.
-   Published benchmarks do not replace testing on the team's hardware
    and data.
-   If the dashboard reads precomputed JSONL/video, describe it as a
    precomputed-results demo rather than live end-to-end inference.

## Tools and technologies

Python · OpenCV · nuScenes devkit · FFmpeg · Ultralytics YOLO11 ·
ByteTrack · PyTorch · NumPy · Streamlit · Git/GitHub

**Team:** Aishwarya Sukesh Nambiar · Harini · Parvati Nair\
**Milestone:** Review 2

from __future__ import annotations

import json
from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st

from risk_layer import WEIGHTS, score_frame
from vp_ui.console import render_console
from vp_ui.payload import build_payload
from vp_ui.tables import alert_list, factor_bars, track_table
from vp_ui.theme import LOGO, shell_css

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "nuscenes"

st.set_page_config(page_title="VisionPriority AI", page_icon=None, layout="wide")


# ---------------------------------------------------------------- data
@st.cache_data(show_spinner="Loading tracked frames")
def load_frames(path: str, mtime: float):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


@st.cache_data(show_spinner="Scoring every frame for live playback")
def load_payload(scene: str, path: str, mtime: float, fps: float):
    return build_payload(scene, load_frames(path, mtime), fps)


def load_meta(scene: str) -> dict:
    p = RESULTS / scene / "meta.json"
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


# ---------------------------------------------------------------- scenes
scenes = sorted(p.name for p in RESULTS.glob("scene*") if (p / "tracks.jsonl").exists())
if not scenes:
    st.error("No results/nuscenes/sceneXX/tracks.jsonl files found. Keep this app.py in the project root.")
    st.stop()

# ---------------------------------------------------------------- sidebar
st.session_state.setdefault("light_mode", st.query_params.get("theme") == "light")
with st.sidebar:
    st.markdown(f'<div class="vp-side-brand">{LOGO}<b>VisionPriority</b></div>', unsafe_allow_html=True)
    scene = st.selectbox("Driving scene", scenes, index=scenes.index("scene01") if "scene01" in scenes else 0)
    light = st.toggle("Light mode", key="light_mode")
mode = "light" if light else "dark"
st.query_params["theme"] = mode
st.markdown(shell_css(mode), unsafe_allow_html=True)

jsonl_path = RESULTS / scene / "tracks.jsonl"
video = RESULTS / scene / "surround_tracks.mp4"
meta = load_meta(scene)
fps = float(meta.get("config", {}).get("fps", 12.0))
frames = load_frames(str(jsonl_path), jsonl_path.stat().st_mtime)

lat = meta.get("latency_ms", {})
trk = meta.get("tracks", {})
det = Path(meta.get("config", {}).get("detector", {}).get("weights", "yolo11s")).stem or "yolo11s"
with st.sidebar:
    st.markdown("---")
    rows = [("Frames", len(frames)), ("Duration", f"{len(frames) / fps:.1f} s"), ("Frame rate", f"{fps:g} fps")]
    if trk.get("unique_ids") is not None:
        rows.append(("Unique tracks", trk["unique_ids"]))
    if lat.get("detect_4views_median") is not None:
        rows.append(("Detect, 4 views", f"{lat['detect_4views_median']:.0f} ms"))
        rows.append(("Device", lat.get("device", "n/a")))
    st.markdown('<div class="vp-kv">' + "".join(f"<span>{escape(str(k))}</span><b>{escape(str(v))}</b>" for k, v in rows) + "</div>",
                unsafe_allow_html=True)

# ---------------------------------------------------------------- masthead
st.markdown(f"""
<div class="vp-mast">
  <div class="vp-brand">{LOGO}
    <div><h1>VisionPriority AI</h1><p>Surround-view hazard awareness for the driver</p></div></div>
  <div class="vp-facts">
    <div class="vp-fact"><b>{escape(scene)}</b><span>Recorded scene replay</span></div>
    <div class="vp-fact"><b>{escape(det.upper() if det.lower().startswith('yolo') else det)} + ByteTrack</b><span>Precomputed tracks</span></div>
    <div class="vp-fact"><b>4 camera views</b><span>Front, front-left, front-right, rear</span></div>
  </div>
</div>
<div class="vp-note">Risk scores are transparent heuristics built from image-plane cues. They are not calibrated collision probabilities,
and the driver advice shown here is a prototype aid, not a safety system.</div>
""", unsafe_allow_html=True)

tab_live, tab_inspect, tab_method = st.tabs(["Live console", "Frame inspector", "How the score works"])

# ---------------------------------------------------------------- live console
with tab_live:
    if not video.exists():
        st.warning("Annotated surround_tracks.mp4 not found for this scene.")
    else:
        payload = load_payload(scene, str(jsonl_path), jsonl_path.stat().st_mtime, fps)
        render_console(payload, ROOT, scene, video)
        st.caption("Press play. The advisory banner, threat map, priority feed and escalation log update on every video frame. "
                   "Select any point on the timeline, or any log entry, to jump to that moment.")

# ---------------------------------------------------------------- frame inspector
with tab_inspect:
    c1, c2 = st.columns([2, 1], gap="large")
    with c1:
        frame_idx = st.slider("Frame", 0, max(0, len(frames) - 1), min(80, len(frames) - 1))
    with c2:
        min_priority = st.selectbox("Show alerts", ["All", "Medium and above", "High / critical only"])

    frame = frames[frame_idx]
    all_scored = score_frame(frame)
    tracks = all_scored
    if min_priority == "Medium and above":
        tracks = [t for t in tracks if t["priority"] != "LOW"]
    elif min_priority == "High / critical only":
        tracks = [t for t in tracks if t["priority"] == "HIGH / CRITICAL"]
    timing = frame.get("timing_ms", {})

    st.markdown(f"""<div class="vp-tiles">
      <div class="vp-tile"><span>Scene</span><b>{escape(scene)}</b></div>
      <div class="vp-tile"><span>Frame</span><b>{frame.get('frame_idx', frame_idx)} / {len(frames) - 1}</b></div>
      <div class="vp-tile"><span>Timestamp</span><b>{frame.get('timestamp', 0):.2f} s</b></div>
      <div class="vp-tile"><span>Active tracks</span><b>{len(frame.get('tracks', []))}</b></div>
    </div>""", unsafe_allow_html=True)

    left, right = st.columns([1.55, 1], gap="large")
    with left:
        st.subheader("Annotated frame")
        if video.exists():
            import cv2
            cap = cv2.VideoCapture(str(video))
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ok, image = cap.read()
            cap.release()
            if ok:
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                cap_txt = f"Surround frame {frame_idx}. Alert values for this frame are listed alongside."
                try:
                    st.image(rgb, width="stretch", caption=cap_txt)
                except TypeError:  # older Streamlit
                    st.image(rgb, use_container_width=True, caption=cap_txt)
            else:
                st.caption("Could not seek to this frame. Use the live console to play the full clip.")
        else:
            st.warning("Annotated surround_tracks.mp4 not found for this scene.")
    with right:
        st.subheader("Ranked alerts")
        st.markdown(alert_list(tracks), unsafe_allow_html=True)
        if tracks:
            st.subheader("Why this score")
            chosen = st.selectbox("Inspect track", tracks, key="inspect",
                                  format_func=lambda x: f"{x['display_id']}  {x['class']}  {x['risk_score']:.2f}")
            st.markdown(f"<div class='vp-note' style='margin:.1rem 0 .4rem'>{escape(chosen['explanation'])}</div>", unsafe_allow_html=True)
            st.markdown(factor_bars(chosen, WEIGHTS), unsafe_allow_html=True)

    st.subheader("Track details")
    if all_scored:
        st.markdown(track_table(all_scored), unsafe_allow_html=True)
        display = pd.DataFrame([{
            "Rank": i + 1, "Priority": t["priority"], "Risk": t["risk_score"], "ID": t["display_id"],
            "View": t["view_display"], "Class": t["class"], "Confidence": t.get("confidence", 0),
            "Zone": t["zone"], "C": t["C"], "Z": t["Z"], "A": t["A"], "D": t["D"], "P": t["P"],
            "Explanation": t["explanation"]} for i, t in enumerate(all_scored)])
        st.download_button("Download scored frame as CSV", display.to_csv(index=False).encode("utf-8"),
                           file_name=f"{scene}_frame_{frame_idx}_risk.csv", mime="text/csv")
    else:
        st.markdown('<div class="vp-empty">No tracked objects in this frame.</div>', unsafe_allow_html=True)
    st.caption(f"Perception timing reported by the tracker: detect {timing.get('detect', 'N/A')} ms, "
               f"track {timing.get('track', 'N/A')} ms, total {timing.get('total', 'N/A')} ms. This excludes Streamlit rendering.")

# ---------------------------------------------------------------- method
with tab_method:
    st.markdown('<div class="vp-formula">Risk = 0.25 C + 0.20 Z + 0.30 A + 0.15 D + 0.10 P</div>', unsafe_allow_html=True)
    st.markdown("""<div class="vp-grid2">
<div class="vp-card"><b>C  Class vulnerability</b><p>Pedestrians and cyclists score highest, then trucks and buses, then cars.</p></div>
<div class="vp-card"><b>Z  Zone criticality</b><p>Image position used as a proximity proxy: how low the box sits and how close it is to the relevant lane corridor for that camera. It is not metric distance.</p></div>
<div class="vp-card"><b>A  Approach rate</b><p>How fast the bounding-box area grows over the last second. An area that doubles each second is treated as full approach.</p></div>
<div class="vp-card"><b>D  Direction alignment</b><p>Motion of the box centre toward the vehicle path, interpreted per camera. This is an image-plane heuristic, so approach rate remains the stronger signal.</p></div>
<div class="vp-card"><b>P  Persistence</b><p>Tracks that have been seen for a full second and matched on most frames score higher than brief or flickering ones.</p></div>
<div class="vp-card"><b>Priority levels</b><p>Low below 0.35, medium from 0.35, high / critical from 0.65. The live console holds the strongest alert for about 0.4 s so the banner does not flicker.</p></div>
</div>""", unsafe_allow_html=True)
    

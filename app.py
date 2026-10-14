from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import streamlit as st
from risk_layer import score_frame

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "nuscenes"
st.set_page_config(page_title="VisionPriority AI", page_icon="🚘", layout="wide")
st.markdown("""
<style>
[data-testid="stSidebar"] { min-width: 205px !important; max-width: 205px !important; }
[data-testid="stSidebar"] > div { padding-top: 1rem; }
.block-container { max-width: 100% !important; padding: 1.25rem 1.5rem 2rem 1.5rem !important; }
h1 { color: #f4f6ff !important; letter-spacing: -0.025em; }
h2, h3 { color: #edf1ff !important; }
[data-testid="stMetric"] {
  background: linear-gradient(135deg, #202b52, #26365f) !important;
  border: 1px solid #3e507f !important; border-radius: 13px !important;
  padding: 15px 17px !important; min-height: 108px;
}
[data-testid="stMetricLabel"] p { color: #c6d2f4 !important; font-size: .9rem !important; }
[data-testid="stMetricValue"] { color: #ffffff !important; font-size: 1.75rem !important; font-weight: 750 !important; }
[data-testid="stMetricDelta"] { color: #d5defa !important; }
[data-testid="stAlert"] { border-radius: 12px; }
div[data-testid="stSelectbox"] label, div[data-testid="stSlider"] label { font-weight: 650; }
div[data-testid="stDataFrame"] { border: 1px solid #303a54; border-radius: 10px; overflow: hidden; }
.alert { background: #171d2b; border: 1px solid #2c3650; border-left: 5px solid #d63a3a;
  padding: 13px 15px; margin: 8px 0; border-radius: 10px; box-shadow: 0 3px 12px #00000018; }
.small { color: #b5bfd8; font-size: .85rem; }
[data-testid="stVideo"] { width: 100%; }
section[data-testid="stSidebar"] hr { margin: .6rem 0; }
@media (max-width: 900px) {
  .block-container { padding: 1rem !important; }
  [data-testid="stSidebar"] { min-width: 185px !important; max-width: 185px !important; }
}
</style>
""", unsafe_allow_html=True)
st.title("🚘 VisionPriority AI")
st.caption("360° SITUATIONAL AWARENESS  ·  DETECT → TRACK → PRIORITIZE → EXPLAIN")
st.caption("DEMO MODE  ·  Precomputed YOLO11s + ByteTrack tracks  ·  Transparent heuristic scores, not calibrated collision probabilities")

scenes = sorted(p.name for p in RESULTS.glob("scene*") if (p/"tracks.jsonl").exists())
if not scenes:
    st.error("No results/nuscenes/sceneXX/tracks.jsonl files found. Keep this app.py in the project root.")
    st.stop()
scene = st.sidebar.selectbox("Choose driving scene", scenes, index=scenes.index("scene01") if "scene01" in scenes else 0)
jsonl_path = RESULTS / scene / "tracks.jsonl"
@st.cache_data(show_spinner="Loading tracked frames…")
def load_frames(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
frames = load_frames(str(jsonl_path))
frame_idx = st.sidebar.slider("Frame / timestamp", 0, max(0,len(frames)-1), min(80,len(frames)-1))
min_priority = st.sidebar.selectbox("Show alerts", ["All", "MEDIUM and above", "HIGH / CRITICAL only"])
frame = frames[frame_idx]
tracks = score_frame(frame)
if min_priority == "MEDIUM and above":
    tracks = [t for t in tracks if t["priority"] != "LOW"]
elif min_priority == "HIGH / CRITICAL only":
    tracks = [t for t in tracks if t["priority"] == "HIGH / CRITICAL"]
timing = frame.get("timing_ms", {})
a,b,c,d = st.columns(4)
a.metric("Scene", scene)
b.metric("Frame", f"{frame.get('frame_idx', frame_idx)} / {len(frames)-1}")
c.metric("Timestamp", f"{frame.get('timestamp',0):.2f} s")
d.metric("Active tracks", len(frame.get("tracks",[])))
st.divider()
left, right = st.columns([1.55, 1], gap="large")
with left:
    st.subheader("Surround-view video")
    video = RESULTS / scene / "surround_tracks.mp4"
    if video.exists():
        import cv2
        cap = cv2.VideoCapture(str(video))
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, image = cap.read()
        cap.release()
        if ok:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            st.image(image, caption=f"Annotated surround frame {frame_idx} — selected frame's alert values are shown alongside", use_container_width=True)
        else:
            st.caption("Could not seek to this frame; use the full clip below.")
        with st.expander("Play the complete annotated scene"):
            st.video(str(video))
    else:
        st.warning("Annotated surround_tracks.mp4 not found for this scene.")
    st.subheader("Surround awareness map")
    # Schematic map: camera direction and highest-risk active object per view.
    view_order = [("CAM_FRONT_LEFT","FRONT-LEFT"),("CAM_FRONT","FRONT"),("CAM_FRONT_RIGHT","FRONT-RIGHT"),("CAM_BACK","REAR")]
    by_view = {}
    for t in score_frame(frame):
        by_view.setdefault(t.get("view",""), t)
    mapcols = st.columns(3)
    with mapcols[0]:
        st.markdown("**FRONT-LEFT**")
        t=by_view.get("CAM_FRONT_LEFT")
        st.write(f"{t['priority']} · {t['display_id']} · {t['risk_score']:.2f}" if t else "No active track")
    with mapcols[1]:
        st.markdown("**FRONT**")
        t=by_view.get("CAM_FRONT")
        st.write(f"{t['priority']} · {t['display_id']} · {t['risk_score']:.2f}" if t else "No active track")
        st.markdown("### 🚘 VEHICLE")
        t=by_view.get("CAM_BACK")
        st.write(f"**REAR:** {t['priority']} · {t['display_id']} · {t['risk_score']:.2f}" if t else "**REAR:** No active track")
    with mapcols[2]:
        st.markdown("**FRONT-RIGHT**")
        t=by_view.get("CAM_FRONT_RIGHT")
        st.write(f"{t['priority']} · {t['display_id']} · {t['risk_score']:.2f}" if t else "No active track")
with right:
    st.subheader("Prioritized alert feed")
    if not tracks:
        st.success("No alerts match the selected priority filter for this frame.")
    high_present = any(t["priority"] == "HIGH / CRITICAL" for t in tracks)
    audio_enabled = st.checkbox("Enable prototype audio alert", value=False)
    beep_path = ROOT / "assets" / "alert_beep.wav"
    if audio_enabled and high_present and beep_path.exists():
        st.warning("High-priority track present — prototype audio cue")
        st.audio(str(beep_path), format="audio/wav", autoplay=True)
    for rank,t in enumerate(tracks[:15],1):
        color = "#d63a3a" if t["priority"] == "HIGH / CRITICAL" else ("#d99000" if t["priority"] == "MEDIUM" else "#2d8a57")
        st.markdown(f"""<div class="alert" style="border-left-color:{color}">
        <b>#{rank} · {t['priority']} · {t['risk_score']:.2f}</b><br>
        {t['explanation']}<br><span class="small">{t['view_display']} · ID {t['display_id']} · {t['class']} · conf {t.get('confidence',0):.2f}</span>
        </div>""", unsafe_allow_html=True)
    if tracks:
        with st.expander("Why this score? — factor breakdown"):
            chosen = st.selectbox("Inspect track", tracks, format_func=lambda x: f"{x['display_id']} | {x['class']} | {x['risk_score']:.2f}", key="inspect")
            st.write(f"**Explanation:** {chosen['explanation']}")
            st.dataframe(pd.DataFrame([{"Factor":"C — Class vulnerability","Score":chosen["C"],"Weight":0.25},
                                       {"Factor":"Z — Zone criticality","Score":chosen["Z"],"Weight":0.20},
                                       {"Factor":"A — Approach rate","Score":chosen["A"],"Weight":0.30},
                                       {"Factor":"D — Direction alignment","Score":chosen["D"],"Weight":0.15},
                                       {"Factor":"P — Persistence","Score":chosen["P"],"Weight":0.10}]), hide_index=True, use_container_width=True)
            st.caption("Risk = 0.25C + 0.20Z + 0.30A + 0.15D + 0.10P")
st.divider()
st.subheader("Track details")
all_scored = score_frame(frame)
if all_scored:
    display = pd.DataFrame([{
        "Rank":i+1,"Priority":t["priority"],"Risk":t["risk_score"],"ID":t["display_id"],
        "View":t["view_display"],"Class":t["class"],"Confidence":t.get("confidence",0),
        "Zone":t["zone"],"C":t["C"],"Z":t["Z"],"A":t["A"],"D":t["D"],"P":t["P"],
        "Explanation":t["explanation"]
    } for i,t in enumerate(all_scored)])
    st.dataframe(display, use_container_width=True, hide_index=True)
    st.download_button("Download scored frame as CSV", display.to_csv(index=False).encode("utf-8"), file_name=f"{scene}_frame_{frame_idx}_risk.csv", mime="text/csv")
else:
    st.write("No tracked objects in this frame.")
st.caption(f"Perception timing reported by tracker: detect {timing.get('detect','N/A')} ms · track {timing.get('track','N/A')} ms · total {timing.get('total','N/A')} ms. This does not include Streamlit rendering.")

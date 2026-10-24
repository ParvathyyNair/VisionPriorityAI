"""Embeds the live console (video + frame-synchronised alerts) in Streamlit."""
from __future__ import annotations

import base64
import json
import shutil
from pathlib import Path

import streamlit as st

from .fonts import font_face_css

TEMPLATE = Path(__file__).with_name("console.html")


def static_video(root: Path, scene: str, src: Path) -> str | None:
    """Copy the scene video into ./static (served at /app/static) and return its file name."""
    try:
        if not bool(st.get_option("server.enableStaticServing")):
            return None
        dst_dir = root / "static"
        dst_dir.mkdir(exist_ok=True)
        name = f"{scene}_{src.name}"   # surround_tracks.mp4 (4 cameras) or surround6_tracks.mp4 (6 cameras)
        dst = dst_dir / name
        if not dst.exists() or dst.stat().st_size != src.stat().st_size:
            shutil.copyfile(src, dst)
        return name
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def _data_uri(path: str, mtime: float) -> str:
    return "data:video/mp4;base64," + base64.b64encode(Path(path).read_bytes()).decode("ascii")


def render_console(payload: dict, root: Path, scene: str, video: Path, height: int = 850) -> None:
    static_name = static_video(root, scene, video)
    data_uri = None if static_name else _data_uri(str(video), video.stat().st_mtime)
    blob = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    boot = (
        f"window.__VP_PAYLOAD__={blob};"
        f"window.__VP_STATIC__={json.dumps(static_name)};"
        f"window.__VP_VIDEO__={json.dumps(data_uri)};"
    )
    html = (TEMPLATE.read_text(encoding="utf-8").replace("/*__PAYLOAD__*/", boot)
            .replace("/*__FONTS__*/", font_face_css()))   # fonts embedded: no internet needed
    if hasattr(st, "iframe"):          # current Streamlit
        st.iframe(html, height=height)
    else:                               # older Streamlit (>=1.35)
        import streamlit.components.v1 as components
        components.html(html, height=height, scrolling=True)

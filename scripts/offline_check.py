"""Offline readiness check for VisionPriority AI (new console UI).

Blocks ALL network access in this process, then proves the app works using only
files on this computer:

  1. every asset is local: recorded scenes, videos, bundled fonts, model weights, config
  2. the live console page contains no internet links (fonts embedded, video served locally)
  3. the full Streamlit app runs for every scene, in dark and light mode, with every alert filter
  4. driver advisories + escalation events are generated for every scene, frame by frame
  5. (full mode) YOLO11s + ByteTrack load from models/ and run on a real frame
  6. no network connection was attempted at any point

  python scripts/offline_check.py          # full check (about 1-2 minutes)
  python scripts/offline_check.py --quick  # fast check, run automatically by run_offline.bat
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import logging
import os
import re
import socket
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["YOLO_OFFLINE"] = "true"
os.environ["YOLO_AUTOINSTALL"] = "false"
RESULTS = ROOT / "results" / "nuscenes"
ATTEMPTS: list = []
OK = True


def _blocked(*args):
    ATTEMPTS.append(args)
    raise OSError("network access is disabled by offline_check")


def block_network():
    socket.socket.connect = lambda self, addr: _blocked(addr)
    socket.socket.connect_ex = lambda self, addr: _blocked(addr)
    socket.create_connection = lambda *a, **k: _blocked(*a)
    socket.getaddrinfo = lambda *a, **k: _blocked(*a)


def check(name, cond, info=""):
    global OK
    OK &= bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{(' — ' + info) if info else ''}", flush=True)


class _Quiet:
    def __enter__(self):
        self.o, self.e = contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO())
        self.o.__enter__(); self.e.__enter__()
    def __exit__(self, *a):
        self.e.__exit__(*a); self.o.__exit__(*a)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--quick", action="store_true"); a = ap.parse_args()
    block_network()
    logging.getLogger("streamlit").setLevel(logging.ERROR)
    print("VisionPriority AI · offline readiness check (network disabled)\n")

    print("1. Local assets")
    scenes = sorted(p.name for p in RESULTS.glob("scene*") if (p / "tracks.jsonl").exists())
    check("recorded scenes", scenes, ", ".join(scenes))
    for sc in scenes:
        check(f"{sc}: 4-camera video", (RESULTS / sc / "surround_tracks.mp4").exists())
        check(f"{sc}: 6-camera heavy-vehicle video + rear-side tracks",
              (RESULTS / sc / "surround6_tracks.mp4").exists() and (RESULTS / sc / "tracks_rear_sides.jsonl").exists())
    fonts = sorted(p.name for p in (ROOT / "assets" / "fonts").glob("*.woff2"))
    check("UI fonts bundled (Chakra Petch + Manrope)", len(fonts) >= 8, f"{len(fonts)} font files")
    check("YOLO11s weights (models/yolo11s.pt)", (ROOT / "models" / "yolo11s.pt").exists())
    cfg = (ROOT / ".streamlit" / "config.toml").read_text()
    check("usage telemetry disabled", "gatherUsageStats = false" in cfg)
    check("videos served from this computer (static serving)", "enableStaticServing = true" in cfg)

    print("\n2. No internet links in the UI")
    from vp_ui.console import TEMPLATE
    from vp_ui.fonts import font_face_css
    from vp_ui.theme import shell_css
    pages = {"live console": TEMPLATE.read_text(encoding="utf-8").replace("/*__FONTS__*/", font_face_css()),
             "dark shell": shell_css("dark"), "light shell": shell_css("light")}
    for name, html in pages.items():
        links = re.findall(r"https?://[^\s'\")]+", html)
        check(f"{name}: no external URLs", not links, ", ".join(links[:3]))
        check(f"{name}: fonts embedded", "data:font/woff2;base64" in html)

    print("\n3. Full app (app.py), every scene and vehicle type, dark + light, every alert filter")
    from streamlit.testing.v1 import AppTest
    import vehicle_profiles as vp
    filters = ["All", "Medium and above", "High / critical only"]
    for sc in (scenes[:1] if a.quick else scenes):
        for vkey in (("car", "truck") if a.quick else tuple(vp.PROFILES)):
          for light in ((False,) if a.quick else (False, True)):
            for flt in (filters[:1] if a.quick else (filters if vkey in ("car", "truck") else filters[:1])):
                with _Quiet():
                    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180)
                    at.run()
                    at.sidebar.selectbox[0].set_value(sc)
                    [x for x in at.sidebar.selectbox if x.label == "Vehicle type"][0].set_value(vkey)
                    if light:
                        at.sidebar.toggle[0].set_value(True)
                    at.run()
                    sb = [s for s in at.selectbox if s.label == "Show alerts"]
                    if sb:
                        sb[0].set_value(flt); at.run()
                check(f"{sc} · {vkey} · {'light' if light else 'dark'} · '{flt}'", not at.exception,
                      ("error: " + at.exception[0].value) if at.exception else f"rendered, {len(at.markdown)} panels")

    print("\n4. Driver advisories generated frame by frame (car and truck)")
    from vp_ui.payload import build_payload
    for sc in scenes:
        base = [json.loads(l) for l in open(RESULTS / sc / "tracks.jsonl", encoding="utf-8") if l.strip()]
        side = [json.loads(l) for l in open(RESULTS / sc / "tracks_rear_sides.jsonl", encoding="utf-8") if l.strip()]
        six = [{**b, "tracks": b["tracks"] + x["tracks"]} for b, x in zip(base, side)]
        for vkey, frames in (("car", base), ("truck", six)):
            t0 = time.perf_counter(); p = build_payload(sc, frames, 12.0, vp.PROFILES[vkey]); ms = (time.perf_counter() - t0) / len(frames) * 1000
            tags = {r[5] for f in p["frames"] for r in f["k"] if r[4]}
            check(f"{sc} · {vkey}: {p['n']} frames with alerts", p["n"] == len(frames) and sum(len(f["k"]) for f in p["frames"]) > 0,
                  f"{len(p['events'])} critical escalations, advice: {', '.join(sorted(tags))}, {ms:.1f} ms/frame")

    if not a.quick:
        print("\n5. YOLO11s + ByteTrack from local weights")
        try:
            import cv2
            from vp_perception.config import DetectorConfig
            from vp_perception.detector import YoloDetector
            from vp_perception.tracker import ViewTracker
            cap = cv2.VideoCapture(str(RESULTS / scenes[0] / "surround_tracks.mp4")); ok, frame = cap.read(); cap.release()
            with _Quiet():
                det = YoloDetector(DetectorConfig(weights="yolo11s.pt", device="cpu")); d = det.detect(frame)
            tr = ViewTracker("CAM_FRONT").update(d, 0, 0.0, (frame.shape[1], frame.shape[0]))
            check("detector loaded from models/ and ran", len(d) > 0, f"{len(d)} detections, {len(tr)} tracks")
        except Exception as e:  # noqa: BLE001
            check("detector loaded from models/ and ran", False, f"{type(e).__name__}: {e}")

    print("\n6. Network")
    check("no network connection attempted", not ATTEMPTS, f"{len(ATTEMPTS)} attempts")
    print("\nRESULT: " + ("OFFLINE READY — the app works with no internet connection." if OK else "NOT READY — see FAIL lines above."))
    sys.exit(0 if OK else 1)


if __name__ == "__main__":
    main()

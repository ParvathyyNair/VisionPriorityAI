"""Offline fonts: the UI's typefaces (Chakra Petch + Manrope, SIL Open Font Licence) are
bundled in assets/fonts and embedded as base64 @font-face rules, so the dashboard looks
identical with no internet connection (no Google Fonts / CDN request is ever made)."""
from __future__ import annotations

import base64
from functools import lru_cache
from pathlib import Path

FONT_DIR = Path(__file__).resolve().parents[1] / "assets" / "fonts"
FAMILIES = (("Chakra Petch", "chakra-petch"), ("Manrope", "manrope"))
WEIGHTS = (400, 500, 600, 700)


@lru_cache(maxsize=1)
def font_face_css() -> str:
    rules = []
    for family, stem in FAMILIES:
        for w in WEIGHTS:
            f = FONT_DIR / f"{stem}-latin-{w}-normal.woff2"
            if f.exists():
                b64 = base64.b64encode(f.read_bytes()).decode("ascii")
                rules.append(f"@font-face{{font-family:'{family}';font-style:normal;font-weight:{w};font-display:swap;"
                             f"src:url(data:font/woff2;base64,{b64}) format('woff2');}}")
    return "\n".join(rules)

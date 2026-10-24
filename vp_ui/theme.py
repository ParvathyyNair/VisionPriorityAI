"""Light and dark design tokens and the Streamlit shell stylesheet.

The live console (an embedded page) reads the --vp-mode custom property from this
document, so flipping the sidebar switch re-themes both without reloading the video.
"""
from __future__ import annotations

from .fonts import font_face_css

TOKENS = {
    "dark": dict(bg="#070b0f", surface="#0d1319", surface2="#131b23", line="#22303b", ink="#e3ebf0",
                 mute="#8497a5", accent="#4fd1e8", ok="#3dcb8c", warn="#f2a93b", crit="#ff5a52",
                 on_accent="#06222a"),
    "light": dict(bg="#e9eef2", surface="#f6f8fa", surface2="#ffffff", line="#c3cfd8", ink="#0c1820",
                  mute="#52666f", accent="#0b7285", ok="#17855a", warn="#b8740a", crit="#d6342c",
                  on_accent="#ffffff"),
}

LOGO = """<svg width="34" height="34" viewBox="0 0 34 34" aria-hidden="true">
<path d="M8 1h25v25l-7 7H1V8z" fill="none" stroke="var(--accent)" stroke-width="2"/>
<path d="M9 22l8-12 8 12" fill="none" stroke="var(--ink)" stroke-width="2"/>
<circle cx="17" cy="20" r="2.4" fill="var(--accent)"/></svg>"""


def shell_css(mode: str) -> str:
    t = TOKENS["light" if mode == "light" else "dark"]
    return f"""
<style>
{font_face_css()}
:root {{
  --vp-mode: {mode};
  --bg:{t['bg']}; --surface:{t['surface']}; --surface2:{t['surface2']}; --line:{t['line']};
  --ink:{t['ink']}; --mute:{t['mute']}; --accent:{t['accent']}; --on-accent:{t['on_accent']};
  --ok:{t['ok']}; --warn:{t['warn']}; --crit:{t['crit']};
  --display:'Chakra Petch','Bahnschrift','Segoe UI',system-ui,sans-serif;
  --body:'Manrope','Segoe UI',system-ui,sans-serif;
}}
html, body, .stApp, [data-testid="stAppViewContainer"] {{ background: var(--bg) !important; color: var(--ink); font-family: var(--body); }}
[data-testid="stHeader"] {{ background: transparent !important; }}
[data-testid="stDeployButton"], #MainMenu, footer {{ display: none !important; }}
.block-container {{ max-width: 1500px !important; padding: 1.1rem 1.6rem 3rem !important; }}

/* sidebar */
[data-testid="stSidebar"] {{ background: var(--surface) !important; border-right: 1px solid var(--line); min-width: 250px !important; max-width: 250px !important; }}
[data-testid="stSidebar"] > div:first-child {{ padding-top: .4rem; }}
[data-testid="stSidebar"] hr {{ border-color: var(--line); margin: .9rem 0; }}

/* text */
.stMarkdown, .stMarkdown p, .stMarkdown li, [data-testid="stWidgetLabel"] p, [data-testid="stCaptionContainer"],
[data-testid="stToggle"] p, label, .stCaption {{ color: var(--ink) !important; font-family: var(--body); }}
[data-testid="stCaptionContainer"], .stCaption {{ color: var(--mute) !important; }}
h1, h2, h3, h4 {{ font-family: var(--display) !important; color: var(--ink) !important; letter-spacing: .005em; }}
h3 {{ font-size: 1.05rem !important; font-weight: 600 !important; }}

/* widgets: selectors cover the older BaseWeb widgets and the newer react-aria widgets */
[data-testid="stAppDeployButton"] {{ display: none !important; }}
[data-testid="stHeader"] button, [data-testid="stHeader"] svg, [data-testid="stSidebarCollapseButton"] button,
[data-testid="stSidebarCollapseButton"] svg, [data-testid="stExpandSidebarButton"] svg, [data-testid="stIconMaterial"] {{ color: var(--ink) !important; fill: var(--ink); }}
[data-testid="stSelectbox"] [role="group"], [data-testid="stSelectbox"] [data-baseweb="select"] > div {{ background: var(--surface2) !important; border: 1px solid var(--line) !important; border-radius: 0 !important; color: var(--ink) !important; }}
[data-testid="stSelectbox"] input, [data-testid="stSelectbox"] [data-baseweb="select"] * {{ color: var(--ink) !important; -webkit-text-fill-color: var(--ink) !important; }}
[data-testid="stSelectbox"] svg {{ fill: var(--mute) !important; color: var(--mute) !important; }}
[role="listbox"], [data-baseweb="popover"] > div, [data-baseweb="menu"] {{ background: var(--surface2) !important; border: 1px solid var(--line) !important; border-radius: 0 !important; }}
[role="option"], [data-baseweb="menu"] li {{ color: var(--ink) !important; background: transparent !important; }}
[role="option"]:hover, [role="option"][aria-selected="true"], [role="option"][data-focused="true"], [data-baseweb="menu"] li:hover {{ background: var(--line) !important; }}
[data-testid="stCheckbox"] p, [data-testid="stCheckbox"] label {{ color: var(--ink) !important; }}
[data-testid="stSliderThumbValue"], [data-testid="stSliderTickBar"] *, [data-testid="stSlider"] [data-testid="stThumbValue"] {{ color: var(--mute) !important; font-family: var(--display); }}
[data-testid="stSlider"] [role="slider"] {{ background: var(--accent) !important; border-radius: 0 !important; box-shadow: none !important; }}
[role="tablist"] {{ gap: 6px; border-bottom: 1px solid var(--line); }}
[role="tab"], button[data-baseweb="tab"] {{ background: transparent !important; color: var(--mute) !important; font-family: var(--display); font-weight: 500; padding: 8px 16px; }}
[role="tab"] p, [role="tab"] div {{ color: inherit !important; font-family: var(--display) !important; }}
[role="tab"][aria-selected="true"], button[data-baseweb="tab"][aria-selected="true"] {{ color: var(--ink) !important; font-weight: 600; box-shadow: inset 0 -3px 0 var(--accent); }}
[data-baseweb="tab-highlight"] {{ background: var(--accent) !important; height: 3px !important; }}
[data-baseweb="tab-border"] {{ background: var(--line) !important; }}
[data-testid="stDownloadButton"] button {{ background: transparent !important; color: var(--ink) !important; border: 1px solid var(--accent) !important; border-radius: 0 !important; font-family: var(--display); font-weight: 500; }}
[data-testid="stDownloadButton"] button p {{ color: inherit !important; }}
[data-testid="stDownloadButton"] button:hover {{ background: var(--accent) !important; color: var(--on-accent) !important; }}
[data-testid="stImage"] img {{ border: 1px solid var(--line); }}
[data-testid="stImageCaption"] {{ color: var(--mute) !important; }}
[data-testid="stAlert"] {{ border-radius: 0 !important; }}
:focus-visible {{ outline: 2px solid var(--accent) !important; outline-offset: 2px; }}

/* masthead */
.vp-mast {{ display: flex; align-items: center; justify-content: space-between; gap: 18px; flex-wrap: wrap; padding: .2rem 0 1rem; border-bottom: 1px solid var(--line); margin-bottom: 1rem; }}
.vp-brand {{ display: flex; align-items: center; gap: 14px; }}
.vp-brand h1 {{ margin: 0 !important; padding: 0 !important; font-size: 1.7rem !important; font-weight: 700 !important; line-height: 1.05; }}
.vp-brand p {{ margin: 3px 0 0; color: var(--mute) !important; font-size: .9rem; }}
.vp-facts {{ display: flex; gap: 22px; flex-wrap: wrap; }}
.vp-fact b {{ display: block; font: 600 .95rem var(--display); color: var(--ink); }}
.vp-fact span {{ color: var(--mute); font-size: .78rem; }}
.vp-note {{ color: var(--mute); font-size: .82rem; margin: .1rem 0 .9rem; max-width: 110ch; }}

/* sidebar brand + scene facts */
.vp-side-brand {{ display: flex; align-items: center; gap: 10px; padding: .4rem 0 .2rem; }}
.vp-side-brand b {{ font: 700 1.05rem var(--display); color: var(--ink); }}
.vp-kv {{ display: grid; grid-template-columns: 1fr auto; gap: 6px 10px; font-size: .84rem; }}
.vp-kv span {{ color: var(--mute); }} .vp-kv b {{ font: 600 .85rem var(--display); color: var(--ink); text-align: right; }}

/* readout tiles */
.vp-tiles {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin: .2rem 0 1rem; }}
.vp-tile {{ border: 1px solid var(--line); background: var(--surface); padding: 10px 14px; border-left: 3px solid var(--accent); }}
.vp-tile span {{ display: block; color: var(--mute); font-size: .78rem; }}
.vp-tile b {{ font: 600 1.45rem var(--display); color: var(--ink); font-variant-numeric: tabular-nums; }}

/* alert list */
.vp-alerts {{ border: 1px solid var(--line); background: var(--surface); }}
.vp-alert {{ display: grid; grid-template-columns: 4px 24px 1fr auto; gap: 10px; align-items: center; padding: 10px 14px 10px 0; border-bottom: 1px solid var(--line); }}
.vp-alert:last-child {{ border-bottom: 0; }}
.vp-alert::before {{ content: ""; align-self: stretch; background: var(--ok); grid-column: 1; }}
.vp-alert.p1::before {{ background: var(--warn); }} .vp-alert.p2::before {{ background: var(--crit); }}
.vp-alert .n {{ grid-column: 2; color: var(--mute); font: 500 .8rem var(--display); text-align: right; }}
.vp-alert .b {{ grid-column: 3; }}
.vp-alert .h {{ font-weight: 600; font-size: .9rem; display: flex; gap: 8px; align-items: baseline; flex-wrap: wrap; color: var(--ink); }}
.vp-alert .d {{ color: var(--mute); font-size: .82rem; margin-top: 2px; }}
.vp-alert .r {{ grid-column: 4; font: 600 1.1rem var(--display); color: var(--ink); }}
.pill {{ font: 600 .68rem var(--display); padding: 2px 6px; border: 1px solid currentColor; }}
.pill.p0 {{ color: var(--ok); }} .pill.p1 {{ color: var(--warn); }} .pill.p2 {{ color: var(--crit); }}
.vp-empty {{ border: 1px dashed var(--line); padding: 22px; color: var(--mute); text-align: center; }}

/* factor bars */
.vp-factors {{ display: grid; gap: 10px; margin-top: .4rem; }}
.vp-factors .f {{ display: grid; grid-template-columns: 1fr 48px; gap: 4px 12px; align-items: center; }}
.vp-factors .l {{ grid-column: 1; display: flex; justify-content: space-between; font-size: .82rem; color: var(--ink); }}
.vp-factors .l span:last-child {{ color: var(--mute); }}
.vp-factors .bar {{ grid-column: 1; height: 6px; background: var(--line); }}
.vp-factors .bar i {{ display: block; height: 100%; background: var(--accent); }}
.vp-factors .v {{ grid-column: 2; grid-row: 1 / span 2; font: 600 1rem var(--display); text-align: right; color: var(--ink); }}

/* table */
.vp-tablewrap {{ overflow: auto; border: 1px solid var(--line); max-height: 460px; background: var(--surface); }}
.vp-table {{ width: 100%; border-collapse: collapse; font-size: .82rem; }}
.vp-table th {{ position: sticky; top: 0; background: var(--surface2); text-align: left; padding: 8px 10px; font: 600 .76rem var(--display); color: var(--mute); border-bottom: 1px solid var(--line); white-space: nowrap; }}
.vp-table td {{ padding: 7px 10px; border-bottom: 1px solid var(--line); color: var(--ink); white-space: nowrap; font-variant-numeric: tabular-nums; }}
.vp-table td:last-child {{ white-space: normal; min-width: 260px; }}
.vp-table tr:hover td {{ background: var(--surface2); }}

/* method tab */
.vp-formula {{ font: 600 1.15rem var(--display); border: 1px solid var(--line); border-left: 3px solid var(--accent); background: var(--surface); padding: 14px 18px; margin: .4rem 0 1rem; color: var(--ink); }}
.vp-grid2 {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }}
.vp-card {{ border: 1px solid var(--line); background: var(--surface); padding: 12px 16px; }}
.vp-card b {{ font: 600 .95rem var(--display); color: var(--ink); }}
.vp-card p {{ margin: 4px 0 0; color: var(--mute) !important; font-size: .85rem; line-height: 1.45; }}

@media (max-width: 900px) {{
  .vp-tiles {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
  .vp-grid2 {{ grid-template-columns: 1fr; }}
  .block-container {{ padding: .8rem .9rem 2rem !important; }}
}}
</style>
"""

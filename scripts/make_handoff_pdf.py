"""Render docs/HANDOFF.md to docs/HANDOFF.pdf (needs `markdown` and `playwright`)."""
from pathlib import Path

import markdown
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
md = (ROOT / "docs" / "HANDOFF.md").read_text()
body = markdown.markdown(md, extensions=["tables", "fenced_code"])
css = """
@page { size: A4; margin: 16mm 14mm; }
body { font-family: 'Segoe UI', Roboto, Arial, sans-serif; font-size: 10pt; color: #1b2440; line-height: 1.45; }
h1 { font-size: 19pt; color: #1b2a5c; border-bottom: 3px solid #e0952a; padding-bottom: 6px; }
h2 { font-size: 14pt; color: #1b2a5c; margin-top: 22px; border-bottom: 1px solid #d6dae6; padding-bottom: 3px; page-break-after: avoid; }
h3 { font-size: 11.5pt; color: #2e3f75; margin-top: 16px; page-break-after: avoid; }
table { border-collapse: collapse; width: 100%; margin: 8px 0 12px; font-size: 8.6pt; page-break-inside: avoid; }
th { background: #1b2a5c; color: #fff; text-align: left; padding: 5px 6px; }
td { border-bottom: 1px solid #e3e6ee; padding: 4px 6px; vertical-align: top; }
tr:nth-child(even) td { background: #f6f7fb; }
code { font-family: Consolas, 'Courier New', monospace; font-size: 8.6pt; background: #f1f3f8; padding: 1px 3px; border-radius: 3px; }
pre { background: #f1f3f8; padding: 8px 10px; border-left: 3px solid #e0952a; overflow: hidden; white-space: pre-wrap; font-size: 8pt; page-break-inside: avoid; }
pre code { background: none; padding: 0; }
hr { border: none; border-top: 1px solid #d6dae6; margin: 14px 0; }
strong { color: #111a38; }
"""
html = f"<!doctype html><html><head><meta charset='utf-8'><style>{css}</style></head><body>{body}</body></html>"
(ROOT / "docs" / "_handoff.html").write_text(html)
with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page()
    pg.goto((ROOT / "docs" / "_handoff.html").as_uri())
    pg.pdf(path=str(ROOT / "docs" / "HANDOFF.pdf"), format="A4", print_background=True,
           display_header_footer=True, header_template="<span></span>",
           footer_template="<div style='font-size:7pt;width:100%;text-align:center;color:#888'>VisionPriority AI · Detection &amp; Tracking handoff · <span class='pageNumber'></span>/<span class='totalPages'></span></div>",
           margin={"top": "14mm", "bottom": "16mm", "left": "13mm", "right": "13mm"})
    b.close()
(ROOT / "docs" / "_handoff.html").unlink()
print("docs/HANDOFF.pdf written")

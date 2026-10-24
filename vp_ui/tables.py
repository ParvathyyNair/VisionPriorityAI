"""Small HTML renderers so tables and bars follow the light/dark theme (st.dataframe cannot)."""
from __future__ import annotations

from html import escape

PRIORITY_CLASS = {"LOW": "p0", "MEDIUM": "p1", "HIGH / CRITICAL": "p2"}


def track_table(rows: list[dict]) -> str:
    head = ["Rank", "Priority", "Risk", "ID", "View", "Class", "Conf.", "Zone", "C", "Z", "A", "D", "P", "Explanation"]
    out = ['<div class="vp-tablewrap"><table class="vp-table"><thead><tr>']
    out += [f"<th>{h}</th>" for h in head]
    out.append("</tr></thead><tbody>")
    for i, t in enumerate(rows, 1):
        pc = PRIORITY_CLASS.get(t["priority"], "p0")
        cells = [
            str(i), f'<span class="pill {pc}">{escape(t["priority"])}</span>', f'{t["risk_score"]:.2f}',
            escape(str(t["display_id"])), escape(str(t["view_display"]).replace("_", "-")), escape(str(t["class"])),
            f'{t.get("confidence", 0):.2f}', escape(str(t["zone"])),
            f'{t["C"]:.2f}', f'{t["Z"]:.2f}', f'{t["A"]:.2f}', f'{t["D"]:.2f}', f'{t["P"]:.2f}',
            escape(str(t["explanation"])),
        ]
        out.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
    out.append("</tbody></table></div>")
    return "".join(out)


def alert_list(rows: list[dict], limit: int = 12) -> str:
    if not rows:
        return '<div class="vp-empty">No tracks match the selected filter in this frame.</div>'
    out = ['<div class="vp-alerts">']
    for i, t in enumerate(rows[:limit], 1):
        pc = PRIORITY_CLASS.get(t["priority"], "p0")
        out.append(
            f'<div class="vp-alert {pc}"><div class="n">{i}</div><div class="b">'
            f'<div class="h"><b>{escape(str(t["display_id"]))}</b> {escape(str(t["class"]))}'
            f'<span class="pill {pc}">{escape(t["priority"])}</span></div>'
            f'<div class="d">{escape(str(t["explanation"]))}. {escape(str(t["view_display"]).replace("_", "-"))} view, '
            f'confidence {t.get("confidence", 0):.2f}</div></div>'
            f'<div class="r">{t["risk_score"]:.2f}</div></div>'
        )
    out.append("</div>")
    return "".join(out)


def factor_bars(t: dict, weights: dict) -> str:
    names = {"C": "Class vulnerability", "Z": "Zone criticality", "A": "Approach rate",
             "D": "Direction alignment", "P": "Persistence"}
    out = ['<div class="vp-factors">']
    for k in "CZADP":
        v = float(t[k])
        out.append(
            f'<div class="f"><div class="l"><span>{names[k]}</span><span>weight {weights[k]:.2f}</span></div>'
            f'<div class="bar"><i style="width:{v * 100:.0f}%"></i></div><div class="v">{v:.2f}</div></div>'
        )
    out.append("</div>")
    return "".join(out)

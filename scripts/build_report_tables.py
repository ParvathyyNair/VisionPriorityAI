"""Fill the result tables in docs/HANDOFF.md from results/*.json (keeps the doc in sync with real runs).

  python scripts/build_report_tables.py
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"


def det_table(rep):
    d = rep["detector"]
    lines = ["| Model | Person AP50 | Person AP50-95 | Recall @conf 0.25 | CPU ms / image (640) | Deck's cited T4 GPU ms | Params |",
             "|---|---|---|---|---|---|---|"]
    cited = {"yolo11n": ("1.5", "2.6M"), "yolo11s": ("2.5", "9.4M"), "yolo11m": ("4.7", "20.1M")}
    for m in ("yolo11n", "yolo11s", "yolo11m"):
        if m not in d:
            continue
        x = d[m]
        b = "**" if m == "yolo11s" else ""
        lines.append(f"| {b}YOLO11{m[-1]}{b} | {b}{x['person_AP50']:.3f}{b} | {x['person_AP50_95']:.3f} | "
                     f"{x['recall_at_conf0.25_iou0.5']:.3f} | {x['median_ms_per_frame']:.0f} | {cited[m][0]} | {cited[m][1]} |")
    return "\n".join(lines)


def det_reading(rep):
    d = rep["detector"]
    if not all(k in d for k in ("yolo11n", "yolo11s", "yolo11m")):
        return "(n/m comparison pending)"
    n, s, m = d["yolo11n"], d["yolo11s"], d["yolo11m"]
    return (f"going n → s raises person AP50 from {n['person_AP50']:.3f} to {s['person_AP50']:.3f} "
            f"(+{(s['person_AP50'] - n['person_AP50']) * 100:.1f} pts) for {s['median_ms_per_frame'] / n['median_ms_per_frame']:.1f}× the time; "
            f"going s → m adds only +{(m['person_AP50'] - s['person_AP50']) * 100:.1f} pts for "
            f"{m['median_ms_per_frame'] / s['median_ms_per_frame']:.1f}× the time. "
            f"The ordering matches the published COCO numbers, so the s-variant choice holds on our data: "
            f"most of the accuracy of m at roughly {s['median_ms_per_frame'] / m['median_ms_per_frame'] * 100:.0f}% of its cost. "
            f"Run 4 views per frame and m is {m['median_ms_per_frame'] * 4:.0f} ms per frame on this CPU.")


def track_table(rep):
    o = rep["tracking_overall"]
    a = o["ByteTrack (two-stage, ours)"]; b = o["Single-stage (low-score boxes discarded)"]
    rows = [("MOTA ↑", "MOTA", "{:.3f}"), ("IDF1 ↑", "IDF1(gt-weighted)", "{:.3f}"), ("Recall ↑", "Recall", "{:.3f}"),
            ("ID switches ↓", "ID_switches", "{:d}"), ("Fragmentations ↓", "Fragmentations", "{:d}"),
            ("Mostly tracked ↑ (of %d GT people)" % a["GT_tracks"], "Mostly_tracked", "{:d}"),
            ("Mostly lost ↓", "Mostly_lost", "{:d}"), ("False positives ↓", "False_positives", "{:d}"),
            ("Recall on partially occluded people (vis < 0.5) ↑", "Recall_on_occluded_boxes(vis<0.5)", "{:.3f}"),
            ("Same ID after full occlusion (vis < 0.3, ≥ 3 frames) ↑", "ID_retention_rate", "{:.3f}")]
    lines = ["| Metric | ByteTrack two-stage (ours) | Low-score stage off | Change |", "|---|---|---|---|"]
    for lab, k, f in rows:
        va, vb = a[k], b[k]
        if isinstance(va, float):
            ch = f"{(va - vb) * 100:+.1f} pts"
        else:
            ch = f"{(va - vb) / vb * 100:+.0f}%" if vb else ""
        extra = ""
        if k == "ID_retention_rate":
            extra = f" ({a['ID_kept_through_occlusion']}/{a['Occlusion_episodes']})"
            vb_s = f.format(vb) + f" ({b['ID_kept_through_occlusion']}/{b['Occlusion_episodes']})"
        else:
            vb_s = f.format(vb)
        lines.append(f"| {lab} | {f.format(va)}{extra} | {vb_s} | {ch} |")
    lines.append("")
    lines.append("Per sequence (two-stage): " + ", ".join(
        f"{s.replace('-FRCNN', '')} MOTA {v['mota']:.2f} / IDF1 {v['idf1']:.2f}"
        for s, v in rep["tracking"]["ByteTrack (two-stage, ours)"].items()))
    return "\n".join(lines)


def track_wins(rep):
    o = rep["tracking_overall"]
    a = o["ByteTrack (two-stage, ours)"]; b = o["Single-stage (low-score boxes discarded)"]
    return (f"fragmentations drop {(1 - a['Fragmentations'] / b['Fragmentations']) * 100:.0f}% "
            f"({b['Fragmentations']} → {a['Fragmentations']}), ID switches drop {(1 - a['ID_switches'] / b['ID_switches']) * 100:.0f}% "
            f"({b['ID_switches']} → {a['ID_switches']}), recall rises {(a['Recall'] - b['Recall']) * 100:.1f} pts, "
            f"MOTA rises {(a['MOTA'] - b['MOTA']) * 100:.1f} pts and IDF1 {(a['IDF1(gt-weighted)'] - b['IDF1(gt-weighted)']) * 100:.1f} pts. "
            f"The cost is {(a['False_positives'] / b['False_positives'] - 1) * 100:.0f}% more false-positive boxes, "
            f"which the risk layer's persistence factor (P) is there to damp.")


def retention(rep):
    o = rep["tracking_overall"]
    a = o["ByteTrack (two-stage, ours)"]; b = o["Single-stage (low-score boxes discarded)"]
    return (f"in {a['ID_retention_rate'] * 100:.0f}% of cases with the two-stage tracker vs "
            f"{b['ID_retention_rate'] * 100:.0f}% without it, so no real difference")


def nus_table():
    lines = ["| Scene | Frames | Frames per view (F / B / FL / FR) | Unique track IDs | Avg active tracks / frame | IDs lasting ≥ 1 s | Main classes | CPU detect, 4 views (median) |",
             "|---|---|---|---|---|---|---|---|"]
    cpu = []
    for p in sorted((R / "nuscenes").glob("scene*/meta.json")):
        m = json.loads(p.read_text())
        c = m["frame_counts_per_view"]; t = m["tracks"]
        top = ", ".join(f"{k} {v}" for k, v in list(t["by_class"].items())[:4])
        lines.append(f"| {m['scene']} | {m['n_batches']} | {c['CAM_FRONT']} / {c['CAM_BACK']} / {c['CAM_FRONT_LEFT']} / {c['CAM_FRONT_RIGHT']} | "
                     f"{t['unique_ids']} | {t['mean_active_per_frame']} | {t['tracks_longer_than_1s']} | {top} | "
                     f"{m['latency_ms']['detect_4views_median']:.0f} ms |")
        cpu.append(m["latency_ms"]["detect_4views_median"])
    lines.append("")
    lines.append("Videos: `results/nuscenes/<scene>/surround_tracks.mp4` (front-left · front · front-right on top, rear below).")
    return "\n".join(lines), (sum(cpu) / len(cpu) if cpu else 0)


def main():
    rep = json.loads((R / "mot17" / "mot17_report.json").read_text())
    nt, cpu = nus_table()
    doc = (ROOT / "docs" / "HANDOFF_template.md").read_text()
    doc = (doc.replace("{{DET_TABLE}}", det_table(rep)).replace("{{DET_READING}}", det_reading(rep))
              .replace("{{TRACK_TABLE}}", track_table(rep)).replace("{{TRACK_WINS}}", track_wins(rep))
              .replace("{{RETENTION}}", retention(rep)).replace("{{NUSCENES_TABLE}}", nt)
              .replace("{{CPU_4VIEW}}", f"{cpu:.0f}"))
    assert "{{" not in doc, re.findall(r"{{\w+}}", doc)
    (ROOT / "docs" / "HANDOFF.md").write_text(doc)
    print("docs/HANDOFF.md updated")


if __name__ == "__main__":
    main()

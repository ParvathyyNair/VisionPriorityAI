"""Validate the detector and ByteTrack against MOT17 ground truth.

Part A — detector comparison on our own frames (YOLO11 n / s / m):
    person AP@0.5 and AP@[.5:.95] against MOT17 GT, plus measured CPU/GPU latency.
Part B — tracking (YOLO11s + ByteTrack), full frame rate:
    MOTA, IDF1, ID switches, fragmentations, MT/ML; compared with the same
    tracker with its low-confidence second stage switched off.
Part C — occlusion check (the reason the deck gives for ByteTrack):
    for every GT person who becomes occluded (visibility < 0.3) and re-appears,
    does the same tracker ID come back afterwards?

Ground-truth handling follows MOTChallenge: targets are class 1 (pedestrian)
with flag 1; classes 2, 7, 8, 12 (person on vehicle, static person,
distractor, reflection) and flag-0 rows are "ignore" regions — predictions
that overlap them are neither TP nor FP.

Requires the detection cache from scripts/cache_mot17_detections.py.
"""
from __future__ import annotations

import argparse
import json
import sys
from argparse import Namespace
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vp_perception.detector import Detections  # noqa: E402

IGNORE_CLASSES = {2, 7, 8, 12}


# ---------------------------------------------------------------- utilities
def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a: (N,4) xyxy, b: (M,4) xyxy -> (N,M) IoU."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0]); iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2]); iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + bb[None, :] - inter + 1e-9)


def load_gt(path: Path):
    g = np.loadtxt(path, delimiter=",")
    out = defaultdict(lambda: {"ids": [], "boxes": [], "vis": [], "ign": []})
    for fr, tid, x, y, w, h, flag, cls, vis in g[:, :9]:
        box = [x - 1, y - 1, x - 1 + w, y - 1 + h]  # MOT is 1-based pixels
        if int(cls) == 1 and int(flag) == 1:
            d = out[int(fr)]; d["ids"].append(int(tid)); d["boxes"].append(box); d["vis"].append(vis)
        elif int(cls) in IGNORE_CLASSES or (int(cls) == 1 and int(flag) == 0):
            out[int(fr)]["ign"].append(box)
    res = {}
    for fr, d in out.items():
        res[fr] = {"ids": np.array(d["ids"], int), "boxes": np.array(d["boxes"]).reshape(-1, 4),
                   "vis": np.array(d["vis"]), "ign": np.array(d["ign"]).reshape(-1, 4)}
    return res


def drop_ignored(pred_boxes: np.ndarray, gt_f: dict, thr=0.5) -> np.ndarray:
    """Boolean keep-mask: drop predictions that match an ignore box better than any target."""
    keep = np.ones(len(pred_boxes), bool)
    if len(pred_boxes) == 0 or len(gt_f["ign"]) == 0:
        return keep
    iou_ign = iou_matrix(pred_boxes, gt_f["ign"]).max(1)
    iou_tgt = iou_matrix(pred_boxes, gt_f["boxes"]).max(1) if len(gt_f["boxes"]) else np.zeros(len(pred_boxes))
    keep[(iou_ign >= thr) & (iou_ign > iou_tgt)] = False
    return keep


def find_frame_offset(cache, gt) -> int:
    """Videos have 1 more frame than the GT; find which video frame == GT frame 1."""
    best, best_off = -1, 0
    for off in (0, 1, -1):
        s = []
        for k, vid_idx in enumerate(cache["frame_idx"][:300]):
            fr = int(vid_idx) + 1 - off
            if fr not in gt or len(cache["dets"][k]) == 0 or len(gt[fr]["boxes"]) == 0:
                continue
            d = cache["dets"][k]; d = d[d[:, 4] >= 0.5]
            if len(d):
                s.extend(list(iou_matrix(d[:, :4], gt[fr]["boxes"]).max(1) >= 0.8))
        m = float(np.mean(s)) if s else 0
        if m > best:
            best, best_off = m, off
    return best_off


# ---------------------------------------------------------------- Part A: AP
def average_precision(records, n_gt):
    """records: list of (score, is_tp). COCO 101-point interpolated AP."""
    if n_gt == 0:
        return float("nan")
    r = sorted(records, key=lambda x: -x[0])
    tp = np.cumsum([x[1] for x in r]); fp = np.cumsum([1 - x[1] for x in r])
    rec = tp / n_gt; prec = tp / np.maximum(tp + fp, 1e-9)
    for i in range(len(prec) - 2, -1, -1):
        prec[i] = max(prec[i], prec[i + 1])
    ap = 0.0
    for t in np.linspace(0, 1, 101):
        j = np.searchsorted(rec, t, side="left")
        ap += prec[j] if j < len(prec) else 0
    return ap / 101


def eval_detection(cache, gt, offset, iou_thrs=np.arange(0.5, 0.96, 0.05)):
    recs = {t: [] for t in iou_thrs}
    n_gt = 0
    recall_at_025 = [0, 0]
    for k, vid_idx in enumerate(cache["frame_idx"]):
        fr = int(vid_idx) + 1 - offset
        if fr not in gt:
            continue
        g = gt[fr]
        d = cache["dets"][k]
        d = d[drop_ignored(d[:, :4], g)] if len(d) else d
        d = d[np.argsort(-d[:, 4])] if len(d) else d
        n_gt += len(g["boxes"])
        ious = iou_matrix(d[:, :4], g["boxes"]) if len(d) else np.zeros((0, len(g["boxes"])))
        for t in iou_thrs:
            used = np.zeros(len(g["boxes"]), bool)
            for i in range(len(d)):
                cand = np.where(~used & (ious[i] >= t))[0]
                if len(cand):
                    used[cand[np.argmax(ious[i, cand])]] = True
                    recs[t].append((d[i, 4], 1))
                else:
                    recs[t].append((d[i, 4], 0))
        if len(d):
            hi = ious[d[:, 4] >= 0.25] if len(g["boxes"]) else None
            if hi is not None and len(hi):
                recall_at_025[0] += int((hi.max(0) >= 0.5).sum())
        recall_at_025[1] += len(g["boxes"])
    aps = {round(float(t), 2): average_precision(recs[t], n_gt) for t in iou_thrs}
    return aps, n_gt, recall_at_025


# ---------------------------------------------------------------- Part B: tracking
def run_bytetrack(cache, high=0.25, low=0.10, new=0.25, buffer=30, match=0.8):
    from ultralytics.trackers.byte_tracker import BYTETracker
    bt = BYTETracker(Namespace(track_high_thresh=high, track_low_thresh=low, new_track_thresh=new,
                               track_buffer=buffer, match_thresh=match, fuse_score=True))
    out = {}
    for k, vid_idx in enumerate(cache["frame_idx"]):
        d = cache["dets"][k]
        dets = Detections(d[:, :4].astype(np.float32), d[:, 4].astype(np.float32), d[:, 5].astype(np.float32)) \
            if len(d) else Detections.empty()
        o = bt.update(dets)
        out[int(vid_idx)] = (o[:, :4].astype(float), o[:, 4].astype(int)) if len(o) else (np.zeros((0, 4)), np.zeros(0, int))
    return out


def eval_tracking(tracks, gt, offset, n_video_frames):
    import motmetrics as mm
    acc = mm.MOTAccumulator(auto_id=False)
    gt_vis = {}
    for vid_idx in range(n_video_frames):
        fr = vid_idx + 1 - offset
        if fr < 1 or fr > max(gt):
            continue
        g = gt.get(fr, {"ids": np.zeros(0, int), "boxes": np.zeros((0, 4)), "vis": np.zeros(0), "ign": np.zeros((0, 4))})
        boxes, ids = tracks.get(vid_idx, (np.zeros((0, 4)), np.zeros(0, int)))
        keep = drop_ignored(boxes, g)
        boxes, ids = boxes[keep], ids[keep]
        iou = iou_matrix(g["boxes"], boxes)
        dist = np.where(iou >= 0.5, 1 - iou, np.nan)
        acc.update(g["ids"].tolist(), ids.tolist(), dist, frameid=fr)
        for i, v in zip(g["ids"], g["vis"]):
            gt_vis[(fr, int(i))] = float(v)
    mh = mm.metrics.create()
    names = ["mota", "idf1", "precision", "recall", "num_switches", "num_fragmentations",
             "mostly_tracked", "mostly_lost", "num_unique_objects", "num_false_positives", "num_misses"]
    summ = mh.compute(acc, metrics=names, name="x").iloc[0].to_dict()
    return summ, acc, gt_vis


def occlusion_analysis(acc, gt_vis, occ_thr=0.3, vis_ok=0.5, min_len=3, max_gap_after=30):
    """For each GT person: find occlusion episodes (vis < occ_thr for >= min_len frames) that have a
    matched frame before and after. Retained = same tracker ID before and after."""
    ev = acc.mot_events.reset_index()
    ev = ev[ev["Type"].isin(["MATCH", "SWITCH"])]
    match = {(int(r.FrameId), int(r.OId)): int(r.HId) for r in ev.itertuples()}
    per_obj = defaultdict(list)
    for (fr, oid), v in gt_vis.items():
        per_obj[oid].append((fr, v))
    episodes = retained = 0
    occ_frames = occ_covered = 0
    for oid, seq in per_obj.items():
        seq.sort()
        frames = [f for f, _ in seq]; vis = np.array([v for _, v in seq])
        for f, v in seq:
            if v < vis_ok:
                occ_frames += 1
                occ_covered += (f, oid) in match
        i = 0
        while i < len(seq):
            if vis[i] < occ_thr:
                j = i
                while j < len(seq) and vis[j] < occ_thr:
                    j += 1
                if j - i >= min_len and i > 0 and j < len(seq):
                    before = next((match[(frames[k], oid)] for k in range(i - 1, max(-1, i - 10), -1)
                                   if (frames[k], oid) in match), None)
                    after = next((match[(frames[k], oid)] for k in range(j, min(len(seq), j + max_gap_after))
                                  if (frames[k], oid) in match), None)
                    if before is not None and after is not None:
                        episodes += 1
                        retained += before == after
                i = j
            else:
                i += 1
    return {"occlusion_episodes": episodes, "id_retained": retained,
            "id_retention_rate": round(retained / episodes, 4) if episodes else None,
            "partially_occluded_gt_boxes(vis<0.5)": occ_frames,
            "partially_occluded_recall": round(occ_covered / occ_frames, 4) if occ_frames else None}


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True, help="folder with MOT17-XX-FRCNN_gt.txt")
    ap.add_argument("--cache", default="results/mot17_cache")
    ap.add_argument("--out", default="results/mot17")
    a = ap.parse_args()
    cache_dir, out = Path(a.cache), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    gts = {p.name.split("_gt")[0]: p for p in sorted(Path(a.gt).glob("MOT17-*_gt.txt"))}
    report = {"detector": {}, "tracking": {}, "occlusion": {}, "frame_offsets": {}}

    # ---------- Part A
    for model in ("yolo11n", "yolo11s", "yolo11m"):
        recs_all = {}
        tot_gt = 0; ms = []; r25 = [0, 0]; per_seq = {}
        for seq, gp in gts.items():
            stride = 1 if model == "yolo11s" else 3
            f = cache_dir / f"{model}_{seq}_s3.npz"
            if not f.exists():
                f = cache_dir / f"{model}_{seq}_s1.npz"
            if not f.exists():
                continue
            c = dict(np.load(f, allow_pickle=True))
            if "_s1" in f.name:  # evaluate the same every-3rd-frame subset as n/m for a fair comparison
                sel = [i for i, v in enumerate(c["frame_idx"]) if v % 3 == 0]
                c = {"dets": c["dets"][sel], "frame_idx": c["frame_idx"][sel], "ms": c["ms"]}
            gt = load_gt(gp)
            off = report["frame_offsets"].setdefault(seq, find_frame_offset(c, gt))
            aps, n_gt, rr = eval_detection(c, gt, off)
            per_seq[seq] = round(aps[0.5], 4)
            tot_gt += n_gt; r25[0] += rr[0]; r25[1] += rr[1]
            ms.extend(list(c["ms"][1:]))
            recs_all[seq] = (aps, n_gt)
        if not recs_all:
            continue
        # GT-weighted mean of per-sequence AP
        w = np.array([n for _, n in recs_all.values()], float)
        ap50 = float(np.average([x[0.5] for x, _ in recs_all.values()], weights=w))
        ap5095 = float(np.average([np.mean(list(x.values())) for x, _ in recs_all.values()], weights=w))
        report["detector"][model] = {"person_AP50": round(ap50, 4), "person_AP50_95": round(ap5095, 4),
                                     "recall_at_conf0.25_iou0.5": round(r25[0] / max(r25[1], 1), 4),
                                     "median_ms_per_frame": round(float(np.median(ms)), 1),
                                     "frames_evaluated_gt_boxes": tot_gt, "per_sequence_AP50": per_seq}
        print(model, report["detector"][model], flush=True)

    # ---------- Part B + C
    variants = {
        "ByteTrack (two-stage, ours)": dict(high=0.25, low=0.10, new=0.25),
        "Single-stage (low-score boxes discarded)": dict(high=0.25, low=0.25, new=0.25),
    }
    agg = {k: defaultdict(float) for k in variants}
    occ = {k: defaultdict(float) for k in variants}
    for seq, gp in gts.items():
        f = cache_dir / f"yolo11s_{seq}_s1.npz"
        if not f.exists():
            continue
        c = dict(np.load(f, allow_pickle=True))
        gt = load_gt(gp)
        off = report["frame_offsets"].get(seq, find_frame_offset(c, gt))
        for name, kw in variants.items():
            tr = run_bytetrack(c, **kw)
            summ, acc, gv = eval_tracking(tr, gt, off, int(c["n_frames"]))
            o = occlusion_analysis(acc, gv)
            report["tracking"].setdefault(name, {})[seq] = {k: (round(v, 4) if isinstance(v, float) else int(v))
                                                           for k, v in summ.items()}
            report["occlusion"].setdefault(name, {})[seq] = o
            for k in ("num_switches", "num_fragmentations", "mostly_tracked", "mostly_lost",
                      "num_unique_objects", "num_false_positives", "num_misses"):
                agg[name][k] += summ[k]
            n_gt_boxes = sum(len(g["boxes"]) for g in gt.values())
            agg[name]["gt_boxes"] += n_gt_boxes
            agg[name]["tp"] += summ["recall"] * n_gt_boxes
            agg[name]["idf1_w"] += summ["idf1"] * n_gt_boxes
            for k in ("occlusion_episodes", "id_retained"):
                occ[name][k] += o[k]
            occ[name]["occ_boxes"] += o["partially_occluded_gt_boxes(vis<0.5)"]
            occ[name]["occ_cov"] += (o["partially_occluded_recall"] or 0) * o["partially_occluded_gt_boxes(vis<0.5)"]
            print(seq, name, f"MOTA={summ['mota']:.3f} IDF1={summ['idf1']:.3f} IDsw={summ['num_switches']}",
                  o, flush=True)
    overall = {}
    for name, A in agg.items():
        if not A["gt_boxes"]:
            continue
        mota = 1 - (A["num_misses"] + A["num_false_positives"] + A["num_switches"]) / A["gt_boxes"]
        O = occ[name]
        overall[name] = {
            "MOTA": round(mota, 4), "IDF1(gt-weighted)": round(A["idf1_w"] / A["gt_boxes"], 4),
            "Recall": round(A["tp"] / A["gt_boxes"], 4),
            "ID_switches": int(A["num_switches"]), "Fragmentations": int(A["num_fragmentations"]),
            "Mostly_tracked": int(A["mostly_tracked"]), "Mostly_lost": int(A["mostly_lost"]),
            "GT_tracks": int(A["num_unique_objects"]), "False_positives": int(A["num_false_positives"]),
            "Occlusion_episodes": int(O["occlusion_episodes"]), "ID_kept_through_occlusion": int(O["id_retained"]),
            "ID_retention_rate": round(O["id_retained"] / O["occlusion_episodes"], 4) if O["occlusion_episodes"] else None,
            "Recall_on_occluded_boxes(vis<0.5)": round(O["occ_cov"] / O["occ_boxes"], 4) if O["occ_boxes"] else None,
        }
    report["tracking_overall"] = overall
    (out / "mot17_report.json").write_text(json.dumps(report, indent=2, default=float))
    print(json.dumps(overall, indent=1))


if __name__ == "__main__":
    main()

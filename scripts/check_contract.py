"""Validate a tracks.jsonl file against the Output Contract.

  python scripts/check_contract.py results/nuscenes/scene00/tracks.jsonl

Checks every record has every contract + extra field, bbox/centroid/area are
consistent, track_age is correct, and that a track's age increases by exactly
the frame gap between consecutive appearances. Exit code 1 if anything fails.
Parvati can run this on any output before feeding it to the risk layer.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vp_perception.contract import validate_record  # noqa: E402


def main(path):
    n_rec = n_frames = 0
    problems = []
    last_seen = {}
    with open(path) as f:
        for line in f:
            fr = json.loads(line)
            n_frames += 1
            for t in fr["tracks"]:
                n_rec += 1
                for e in validate_record(t):
                    problems.append(f"frame {t['frame_idx']} {t['track_label']}: {e}")
                key = (t["view"], t["track_id"])
                if key in last_seen:
                    pf, pa = last_seen[key]
                    if t["track_age"] - pa != t["frame_idx"] - pf:
                        problems.append(f"frame {t['frame_idx']} {t['track_label']}: age jumped {pa}->{t['track_age']}")
                last_seen[key] = (t["frame_idx"], t["track_age"])
    print(f"{path}: {n_frames} frames, {n_rec} track records, {len(last_seen)} unique (view, track_id)")
    if problems:
        print(f"FAIL — {len(problems)} problems, first 10:")
        for p in problems[:10]:
            print("  ", p)
        sys.exit(1)
    print("PASS — every record matches the Output Contract")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        main(p)

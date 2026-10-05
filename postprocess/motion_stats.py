#!/usr/bin/env python3
"""Camera-motion self-check for a by-type trial dataset.

For every trajectory JSON (per-frame x/y/z), form frame pairs (i, i+offset)
and report the fraction whose camera displacement is >= min_disp blocks.
Also reports the longest run of consecutive "static" frames (frame-to-frame
displacement < static_eps), which is what a downstream RPE check trips on.

Usage:
  motion_stats.py output/trial/orbit/dataset/trajectories [more dirs...] \
      [--offset 4] [--min-disp 0.5]
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


def load_xyz(path: Path):
    data = json.load(open(path))
    return [(float(e["x"]), float(e["y"]), float(e["z"])) for e in data]


def dist(a, b):
    return math.dist(a, b)


def traj_stats(xyz, offset, min_disp, static_eps, fps):
    n = len(xyz)
    pairs = max(0, n - offset)
    moving = sum(1 for i in range(pairs) if dist(xyz[i], xyz[i + offset]) >= min_disp)
    frac = moving / pairs if pairs else 0.0

    # Longest static run in seconds (consecutive frames with ~zero motion)
    longest = cur = 0
    for i in range(1, n):
        if dist(xyz[i - 1], xyz[i]) < static_eps:
            cur += 1
            longest = max(longest, cur)
        else:
            cur = 0
    static_frac = (
        sum(1 for i in range(1, n) if dist(xyz[i - 1], xyz[i]) < static_eps) / (n - 1)
        if n > 1
        else 0.0
    )
    path_len = sum(dist(xyz[i - 1], xyz[i]) for i in range(1, n))
    return {
        "frames": n,
        "moving_frac": frac,
        "static_frac": static_frac,
        "longest_static_s": longest / fps,
        "path_len": path_len,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dirs", nargs="+", type=Path, help="trajectories/ directories")
    ap.add_argument("--offset", type=int, default=4, help="frame offset (default 4 = 0.2s @20fps)")
    ap.add_argument("--min-disp", type=float, default=0.5, help="min displacement in blocks (default 0.5)")
    ap.add_argument("--static-eps", type=float, default=0.02, help="per-frame motion below this counts as static")
    ap.add_argument("--fps", type=float, default=20.0)
    ap.add_argument("--per-traj", action="store_true", help="print one line per trajectory")
    args = ap.parse_args()

    for d in args.dirs:
        files = sorted(d.glob("*.json"))
        if not files:
            print(f"{d}: no trajectories")
            continue
        rows = []
        for f in files:
            xyz = load_xyz(f)
            if len(xyz) <= args.offset:
                continue
            s = traj_stats(xyz, args.offset, args.min_disp, args.static_eps, args.fps)
            s["name"] = f.stem
            rows.append(s)
            if args.per_traj:
                print(
                    f"  {f.stem:45s} frames={s['frames']:5d} moving={s['moving_frac']*100:5.1f}% "
                    f"static={s['static_frac']*100:5.1f}% longest_static={s['longest_static_s']:5.1f}s "
                    f"path={s['path_len']:6.1f}"
                )
        mf = [r["moving_frac"] for r in rows]
        sf = [r["static_frac"] for r in rows]
        ls = [r["longest_static_s"] for r in rows]
        pl = [r["path_len"] for r in rows]
        # Pooled (frame-weighted) moving fraction as well as per-traj mean
        tot_pairs = sum(r["frames"] - args.offset for r in rows)
        pooled = sum(r["moving_frac"] * (r["frames"] - args.offset) for r in rows) / tot_pairs
        print(
            f"{str(d):55s} n={len(rows):3d} | moving(offset={args.offset},>={args.min_disp}): "
            f"pooled={pooled*100:5.1f}%  mean={statistics.mean(mf)*100:5.1f}%  "
            f"min={min(mf)*100:5.1f}%  max={max(mf)*100:5.1f}% | "
            f"static frames mean={statistics.mean(sf)*100:5.1f}% | "
            f"longest static mean={statistics.mean(ls):4.1f}s max={max(ls):4.1f}s | "
            f"path mean={statistics.mean(pl):5.1f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

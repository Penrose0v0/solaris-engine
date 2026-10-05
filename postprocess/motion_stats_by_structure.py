#!/usr/bin/env python3
"""Per-episode camera-motion check for buildStructure, grouped by structure_type.

Reads <dataset>/episode_info/*.json for `structure_type` and
<dataset>/trajectories/*.json for per-frame pose, and reports for each
trajectory the fraction of (i, i+offset) frame pairs whose camera displacement
is >= min_disp blocks, then summarizes per structure_type.

Usage: motion_stats_by_structure.py <dataset_dir> [--offset 4] [--min-disp 0.5] [--pass 0.5]
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


def moving_frac(xyz, offset, min_disp):
    pairs = len(xyz) - offset
    if pairs <= 0:
        return 0.0
    return sum(1 for i in range(pairs) if math.dist(xyz[i], xyz[i + offset]) >= min_disp) / pairs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dataset", type=Path)
    ap.add_argument("--offset", type=int, default=4)
    ap.add_argument("--min-disp", type=float, default=0.5)
    ap.add_argument("--pass", dest="pass_thr", type=float, default=0.5,
                    help="per-trajectory moving fraction counted as 'healthy' (default 0.5)")
    args = ap.parse_args()

    infos = {}
    for p in (args.dataset / "episode_info").glob("*_episode_info.json"):
        d = json.load(open(p))
        infos[p.name[: -len("_episode_info.json")]] = d.get("structure_type", "<missing>")

    rows = []
    for p in sorted((args.dataset / "trajectories").glob("*.json")):
        stype = infos.get(p.stem, "<no episode_info>")
        d = json.load(open(p))
        xyz = [(float(e["x"]), float(e["y"]), float(e["z"])) for e in d]
        rows.append((p.stem, stype, len(xyz), moving_frac(xyz, args.offset, args.min_disp)))

    print(f"{'trajectory':45s} {'structure':10s} {'frames':>6s} {'moving%':>8s}")
    for name, stype, n, mf in rows:
        print(f"{name:45s} {stype:10s} {n:6d} {mf*100:7.1f}%")

    by = defaultdict(list)
    for _, stype, n, mf in rows:
        by[stype].append((n, mf))
    print()
    print(f"{'structure':10s} {'traj':>4s} {'pooled%':>8s} {'mean%':>7s} {'min%':>6s} {'max%':>6s} {'healthy(>=' + str(int(args.pass_thr*100)) + '%)':>16s}")
    tot_pairs = tot_mov = 0
    for stype in sorted(by):
        lst = by[stype]
        pairs = sum(n - args.offset for n, _ in lst)
        mov = sum(mf * (n - args.offset) for n, mf in lst)
        tot_pairs += pairs; tot_mov += mov
        healthy = sum(1 for _, mf in lst if mf >= args.pass_thr)
        mfs = [mf for _, mf in lst]
        print(f"{stype:10s} {len(lst):4d} {mov/pairs*100:7.1f}% {statistics.mean(mfs)*100:6.1f}% {min(mfs)*100:5.1f}% {max(mfs)*100:5.1f}% {healthy:>7d}/{len(lst):<8d}")
    print(f"{'ALL':10s} {len(rows):4d} {tot_mov/tot_pairs*100:7.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

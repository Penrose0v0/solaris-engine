#!/usr/bin/env python3
"""Estimate the rendered camera's focal length per recording from the GT
rotation, to detect recordings rendered with a boosted FOV.

Minecraft renders the first-person view with vertical FOV 70 deg (focal
length f = (H/2)/tan(35 deg) = 514 px at 720p), but multiplies the FOV by 1.1
while the player is flying (77 deg, f = 452 px). The recording cameras were
put into flight by the MirrorBot plugin (until 2026-10-16), and whether the
client had cleared that state again varied per recording, so some videos
were rendered at 77 deg.

Method: for every pair of consecutive frames whose GT look direction turned
by at least MIN_STEP_DEG, the first frame is warped by the homography of the
GT rotation, H = K R K^-1, for each focal length of a grid, and the focal
length whose warp matches the second frame best (normalised cross
correlation over the central image region, HUD excluded) is the pair's
estimate. Pairs whose best match sits on the grid boundary are dropped (the
aligned video repeats or skips a frame now and then, which looks like a zero
or a double step). The recording's estimate is the median over its pairs.

Input: a flattened dataset root with <type>/{videos,trajectories}; or any
directory tree holding <stem>.mp4 and <stem>.json pairs (--pairs).
Output: <out>.json with per-recording f_est, n_pairs, fov_est_deg, verdict.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

VFOV_DEG = 70.0
FLY_FACTOR = 1.1
MIN_STEP_DEG = 0.6
MAX_PAIRS = 40
F_GRID = np.arange(380.0, 660.0, 6.0)   # focal lengths at full resolution (720p)
MIN_NCC = 0.5
MIN_PAIRS = 3
SCALE = 0.5                             # work at half resolution
DEG = math.pi / 180.0


def focal(height: int, vfov_deg: float) -> float:
    return (height / 2.0) / math.tan(vfov_deg * DEG / 2.0)


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def cam_matrix(yaw: float, pitch: float) -> np.ndarray:
    """Rows (right, down, forward) of the camera frame in world coordinates
    (mineflayer convention: yaw 0 = north = -z, counter-clockwise; pitch
    positive = up), so that pixel x grows right and pixel y grows down."""
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    fwd = (-sy * cp, sp, -cy * cp)
    right = (cy, 0.0, -sy)
    up = (sy * sp, cp, cy * sp)
    return np.array([right, [-u for u in up], fwd], dtype=np.float64)


def estimate(video: Path, frames: list) -> dict:
    cap = cv2.VideoCapture(str(video))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    n = min(n, len(frames))
    cands = []
    for k in range(n - 1):
        a, b = frames[k], frames[k + 1]
        step = math.hypot(wrap(b["yaw"] - a["yaw"]) * math.cos(a["pitch"]), b["pitch"] - a["pitch"])
        if step >= MIN_STEP_DEG * DEG:
            cands.append(k)
    if len(cands) > MAX_PAIRS:
        stride = len(cands) / MAX_PAIRS
        cands = [cands[int(i * stride)] for i in range(MAX_PAIRS)]
    w, h = int(W * SCALE), int(H * SCALE)
    cx, cy = w / 2.0, h / 2.0
    # Central region (HUD and hand excluded) where the match is scored.
    y0, y1, x0, x1 = int(h * 0.12), int(h * 0.72), int(w * 0.15), int(w * 0.85)
    ests, nccs = [], []
    prev_k = None
    for k in cands:
        if prev_k != k:
            cap.set(cv2.CAP_PROP_POS_FRAMES, k)
            ok, a = cap.read()
            if not ok:
                continue
        ok, b = cap.read()
        if not ok:
            break
        prev_k = k + 1
        ga = cv2.resize(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), (w, h), interpolation=cv2.INTER_AREA).astype(np.float32)
        gb = cv2.resize(cv2.cvtColor(b, cv2.COLOR_BGR2GRAY), (w, h), interpolation=cv2.INTER_AREA).astype(np.float32)
        a = b
        Ma = cam_matrix(frames[k]["yaw"], frames[k]["pitch"])
        Mb = cam_matrix(frames[k + 1]["yaw"], frames[k + 1]["pitch"])
        R = Mb @ Ma.T
        tb = gb[y0:y1, x0:x1]
        tb = tb - tb.mean()
        tb_norm = np.linalg.norm(tb) + 1e-6
        best_f, best_ncc = None, -1.0
        scores = []
        for f in F_GRID:
            fs = f * SCALE
            K = np.array([[fs, 0, cx], [0, fs, cy], [0, 0, 1.0]])
            Hm = K @ R @ np.linalg.inv(K)
            warped = cv2.warpPerspective(ga, Hm, (w, h), flags=cv2.INTER_LINEAR, borderValue=-1.0)
            ta = warped[y0:y1, x0:x1]
            valid = ta >= 0
            if valid.mean() < 0.8:
                scores.append(-1.0)
                continue
            ta = np.where(valid, ta, 0.0)
            tbv = np.where(valid, tb, 0.0)
            ta = ta - ta[valid].mean()
            ncc = float((ta * tbv).sum() / ((np.linalg.norm(ta) + 1e-6) * tb_norm))
            scores.append(ncc)
            if ncc > best_ncc:
                best_ncc, best_f = ncc, float(f)
        if best_f is None or best_ncc < MIN_NCC:
            continue
        if best_f <= F_GRID[0] or best_f >= F_GRID[-1]:
            continue  # repeated / double-stepped video frame
        ests.append(best_f)
        nccs.append(best_ncc)
    cap.release()
    out = {"n_pairs": len(ests), "n_candidates": len(cands), "height": H}
    if len(ests) >= MIN_PAIRS:
        ests.sort()
        f = ests[len(ests) // 2]
        q1, q3 = ests[len(ests) // 4], ests[(3 * len(ests)) // 4]
        out["f_est"] = f
        out["f_iqr"] = q3 - q1
        out["ncc_med"] = sorted(nccs)[len(nccs) // 2]
        out["fov_est_deg"] = 2 * math.atan((H / 2.0) / f) / DEG
        f70, f77 = focal(H, VFOV_DEG), focal(H, VFOV_DEG * FLY_FACTOR)
        out["f_nominal"] = f70
        out["f_flying"] = f77
        mid = math.sqrt(f70 * f77)
        if abs(f - f70) <= 0.06 * f70:
            out["verdict"] = "nominal"
        elif abs(f - f77) <= 0.06 * f77:
            out["verdict"] = "boosted"
        else:
            out["verdict"] = "unclear"
        out["side"] = "nominal" if f > mid else "boosted"
    else:
        out["verdict"] = "unknown"
    return out


def iter_pairs(root: Path, pairs: bool, types: set | None):
    if pairs:
        for v in sorted(root.rglob("*.mp4")):
            j = v.with_suffix(".json")
            if j.exists():
                yield v.stem, v, j
        return
    for tdir in sorted(p for p in root.iterdir() if p.is_dir()):
        if types and tdir.name not in types:
            continue
        vd, td = tdir / "videos", tdir / "trajectories"
        if not vd.is_dir() or not td.is_dir():
            continue
        for v in sorted(vd.glob("*.mp4")):
            j = td / f"{v.stem}.json"
            if j.exists():
                yield f"{tdir.name}/{v.stem}", v, j


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--pairs", action="store_true", help="root holds <stem>.mp4 + <stem>.json pairs (any depth)")
    ap.add_argument("--types", default=None, help="comma-separated episode type dirs to include")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    types = set(args.types.split(",")) if args.types else None
    results = {}
    for i, (name, v, j) in enumerate(iter_pairs(args.root, args.pairs, types)):
        if args.limit and i >= args.limit:
            break
        frames = json.loads(j.read_text())
        if not isinstance(frames, list) or not frames or "yaw" not in frames[0]:
            continue
        results[name] = estimate(v, frames)
        r = results[name]
        print(f"{name:70s} pairs {r['n_pairs']:3d}/{r['n_candidates']:3d} f_est {r.get('f_est', float('nan')):6.1f} "
              f"iqr {r.get('f_iqr', float('nan')):5.1f} ncc {r.get('ncc_med', float('nan')):.2f} fov {r.get('fov_est_deg', float('nan')):5.1f} {r['verdict']}", flush=True)
    out = args.out or (args.root / "camera_fov_check.json")
    out.write_text(json.dumps(results, indent=1))
    verdicts = {}
    for name, r in results.items():
        inst = name.split("instance_")[-1][:3] if "instance_" in name else "?"
        verdicts.setdefault(inst, {}).setdefault(r["verdict"], 0)
        verdicts[inst][r["verdict"]] += 1
    print(f"\nwrote {out}")
    for inst, vs in sorted(verdicts.items()):
        print(f"instance {inst}: {vs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

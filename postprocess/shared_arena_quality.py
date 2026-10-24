#!/usr/bin/env python3
"""Per-episode quality manifest and review montages for the sharedArenaEval
dataset (two bots inside a random-colour concrete enclosure, sweeping their
yaw without moving).

Input: a flattened dataset root holding sharedArenaEval/{trajectories,
episode_info,videos}. The two recordings of an episode are paired by episode
number and instance; frames are paired by renderTime inside the motion
window (see co_observe_quality.pair_frames).

Per recording: the position must not change inside the window, the yaw
range / travel / reversals and the measured angular speed, the pitch range,
and the HUD overlay heuristics (toast panel, yellow chat text, chat
darkening) of co_observe_quality.

Per frame pair: the angle between the two view directions and the overlap
of the two views over the enclosure surface (inner wall faces and floor
tops, one sample point per block face): points visible in both views /
visible in either (IoU) and the number visible in both. The first pair
gives the start-view overlap.

Output: <root>/quality_manifest_shared_arena.json, a summary on stdout and
<root>/shared_arena_review/<ep>_<inst>.png (A | B at the window start,
middle and end).
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import re
from pathlib import Path

import cv2
import numpy as np

from co_observe_quality import (Video, angle_between, forward, overlay_scan, pair_frames, pct, visible_many,
                                window_of)

TYPE = "sharedArenaEval"
MIN_WINDOW_FRAMES = 150
POS_EPS = 0.01
MIN_YAW_RANGE_DEG = 45
MIN_START_IOU = 0.05
DEG = 180.0 / math.pi

STEM_RE = re.compile(r"^(?P<prefix>.+?)_(?P<ep>\d{6})_(?P<bot>[A-Za-z]+)_instance_(?P<inst>\d{3})$")


def enclosure_points(info: dict) -> np.ndarray:
    """One point per inner wall face and per floor top."""
    enc = info["enclosure"]
    cx, cy, cz = info["centre"]
    cxi, czi = math.floor(cx), math.floor(cz)
    off = enc["wall_offset"]
    pts = []
    for b in enc["blocks"]:
        x, y, z = b["x"], b["y"], b["z"]
        if b.get("part") == "floor":
            pts.append((x + 0.5, y + 1.0, z + 0.5))
            continue
        dx, dz = x - cxi, z - czi
        if dx == off:
            pts.append((x, y + 0.5, z + 0.5))
        if dx == -off:
            pts.append((x + 1.0, y + 0.5, z + 0.5))
        if dz == off:
            pts.append((x + 0.5, y + 0.5, z))
        if dz == -off:
            pts.append((x + 0.5, y + 0.5, z + 1.0))
    return np.array(pts, dtype=float)


def unwrap(yaws):
    out = [yaws[0]]
    for y in yaws[1:]:
        d = (y - out[-1] + math.pi) % (2 * math.pi) - math.pi
        out.append(out[-1] + d)
    return np.array(out)


def recording_sweep(info: dict, frames: list, win) -> dict:
    out = {"bot": info["bot_name"], "num_frames": len(frames), "window": list(win) if win else None,
           "window_frames": (win[1] - win[0] + 1) if win else None, "plan": info.get("look_plan")}
    if not win:
        return out
    seg = frames[win[0]: win[1] + 1]
    xs, ys, zs = (np.array([f[k] for f in seg]) for k in ("x", "y", "z"))
    out["pos_drift"] = float(max(xs.max() - xs.min(), ys.max() - ys.min(), zs.max() - zs.min()))
    yaw = unwrap([f["yaw"] for f in seg]) * DEG
    pitch = np.array([f["pitch"] for f in seg]) * DEG
    d = np.diff(yaw)
    moving = d[np.abs(d) > 0.05]
    signs = np.sign(moving)
    reversals = int(np.sum(signs[1:] != signs[:-1])) if len(signs) > 1 else 0
    out.update({
        "yaw_start_deg": float(yaw[0]), "yaw_min_deg": float(yaw.min()), "yaw_max_deg": float(yaw.max()),
        "yaw_range_deg": float(yaw.max() - yaw.min()), "yaw_travel_deg": float(np.abs(d).sum()),
        "yaw_reversals": reversals,
        "omega_measured_deg_per_sec": float(np.median(np.abs(moving)) * 20) if len(moving) else 0.0,
        "pitch_min_deg": float(pitch.min()), "pitch_max_deg": float(pitch.max()),
        "keys_pressed": sorted({k for f in seg for k in ("forward", "back", "left", "right", "jump", "sneak", "sprint") if f["action"].get(k)}),
    })
    return out


def analyze_episode(a, b) -> dict:
    (base_a, info_a, fr_a, vid_a), (base_b, info_b, fr_b, vid_b) = a, b
    rec = {
        "type": TYPE,
        "bots": [info_a["bot_name"], info_b["bot_name"]],
        "primary": info_a.get("primary"),
        "world_type": info_a.get("world_type"),
        "scene_check_passed": bool(info_a.get("scene_check_passed")) and bool(info_b.get("scene_check_passed")),
        "centre": info_a.get("centre"),
        "wall_height": (info_a.get("enclosure") or {}).get("wall_height"),
        "start_overlap_planned": info_a.get("start_overlap"),
        "scene_search_attempts": info_a.get("scene_search_attempts") or info_b.get("scene_search_attempts"),
        "encountered_error": bool(info_a.get("encountered_error") or info_b.get("encountered_error")),
        "reasons": [],
    }
    if not rec["scene_check_passed"]:
        rec["reasons"].append("scene search failed")
        rec["clean"] = False
        return rec
    if rec["encountered_error"]:
        rec["reasons"].append("controller error")
    va, vb = Video(vid_a), Video(vid_b)
    wa, wb = window_of(info_a, fr_a), window_of(info_b, fr_b)
    rec["recordings"] = {}
    for info, frames, win, vid in ((info_a, fr_a, wa, va), (info_b, fr_b, wb, vb)):
        m = recording_sweep(info, frames, win)
        m["overlay"] = overlay_scan(vid)
        rec["recordings"][info["bot_name"]] = m
        tag = info["bot_name"]
        if not win or m["window_frames"] < MIN_WINDOW_FRAMES:
            rec["reasons"].append(f"{tag}: window {m['window_frames']} frames < {MIN_WINDOW_FRAMES}")
            continue
        if m["pos_drift"] > POS_EPS:
            rec["reasons"].append(f"{tag}: moved {m['pos_drift']:.2f} blocks")
        if m["yaw_range_deg"] < MIN_YAW_RANGE_DEG:
            rec["reasons"].append(f"{tag}: yaw range {m['yaw_range_deg']:.0f} deg < {MIN_YAW_RANGE_DEG}")
        if m["keys_pressed"]:
            rec["reasons"].append(f"{tag}: movement keys {m['keys_pressed']}")
        # The HUD overlay heuristics (light-grey toast panel top right, yellow
        # chat text bottom left) fire on the white / yellow concrete of the
        # walls, so they stay informational here; the cameras run with chat
        # hidden and the montages are the check.
    pairs = pair_frames(fr_a, fr_b, wa, wb) if wa and wb else []
    rec["num_pairs"] = len(pairs)
    if wa and wb and len(pairs) < MIN_WINDOW_FRAMES:
        rec["reasons"].append(f"only {len(pairs)} paired frames")
    pts = enclosure_points(info_a)
    per = []
    for ka, kb in pairs:
        pa, pb = fr_a[ka], fr_b[kb]
        sa = visible_many(pa, pts, va.width, va.height)
        sb = visible_many(pb, pts, vb.width, vb.height)
        union = int((sa | sb).sum())
        per.append({"ka": ka, "kb": kb, "dt_ms": pa["renderTime"] - pb["renderTime"],
                    "view_angle_deg": angle_between(forward(pa["yaw"], pa["pitch"]), forward(pb["yaw"], pb["pitch"])),
                    "yaw_a_deg": pa["yaw"] * DEG, "yaw_b_deg": pb["yaw"] * DEG,
                    "both": int((sa & sb).sum()), "iou": float((sa & sb).sum() / union) if union else 0.0})
    rec["pairs"] = per
    if per:
        ious = np.array([p["iou"] for p in per])
        rec["start_overlap"] = {"iou": per[0]["iou"], "points_both": per[0]["both"], "view_angle_deg": per[0]["view_angle_deg"]}
        rec["overlap"] = {"iou_median": float(np.median(ious)), "iou_min": float(ious.min()), "iou_max": float(ious.max()),
                          "share_ge_0.3": float((ious >= 0.3).mean()), "share_zero": float((ious == 0).mean())}
        if per[0]["iou"] < MIN_START_IOU:
            rec["reasons"].append(f"start views overlap too little (IoU {per[0]['iou']:.2f})")
    va.close()
    vb.close()
    rec["clean"] = not rec["reasons"]
    return rec


def montage(rec: dict, a, b, out_path: Path):
    (base_a, info_a, fr_a, vid_a), (base_b, info_b, fr_b, vid_b) = a, b
    pairs = rec.get("pairs") or []
    if not pairs:
        return False
    va, vb = Video(vid_a), Video(vid_b)
    picks = [pairs[min(3, len(pairs) - 1)], pairs[len(pairs) // 2], pairs[max(0, len(pairs) - 4)]]
    rows = []
    for p in picks:
        ia, ib = va.frame(p["ka"]), vb.frame(p["kb"])
        if ia is None or ib is None:
            continue
        row = np.concatenate([cv2.resize(ia, (640, 360)), cv2.resize(ib, (640, 360))], axis=1)
        txt = (f"{info_a['bot_name']} f{p['ka']} yaw {p['yaw_a_deg']:.0f} | {info_b['bot_name']} f{p['kb']} yaw {p['yaw_b_deg']:.0f} | "
               f"view angle {p['view_angle_deg']:.0f} deg  overlap IoU {p['iou']:.2f}")
        cv2.putText(row, txt, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(row, txt, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        rows.append(row)
    va.close()
    vb.close()
    if not rows:
        return False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(out_path), np.concatenate(rows, axis=0)))


def load(root: Path):
    info_dir, traj_dir, video_dir = root / TYPE / "episode_info", root / TYPE / "trajectories", root / TYPE / "videos"
    by_episode = {}
    for info_path in sorted(info_dir.glob("*_episode_info.json")):
        base = info_path.name[: -len("_episode_info.json")]
        m = STEM_RE.match(base)
        traj_path = traj_dir / f"{base}.json"
        if not m or not traj_path.exists():
            continue
        info = json.loads(info_path.read_text())
        frames = json.loads(traj_path.read_text())
        by_episode.setdefault((m["inst"], m["ep"]), {})[info["bot_name"]] = (base, info, frames, video_dir / f"{base}.mp4")
    return by_episode


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--no-montage", action="store_true")
    args = ap.parse_args()

    records = []
    review_dir = args.root / "shared_arena_review"
    for (inst, ep), bots in sorted(load(args.root).items()):
        names = sorted(bots)
        if len(names) < 2:
            records.append({"type": TYPE, "episode": ep, "instance": inst, "bots": names, "clean": False,
                            "scene_check_passed": bool(bots[names[0]][1].get("scene_check_passed")),
                            "reasons": ["second recording missing"]})
            continue
        a, b = bots[names[0]], bots[names[1]]
        rec = analyze_episode(a, b)
        rec["episode"] = ep
        rec["instance"] = inst
        rec["recording_files"] = {name: f"{TYPE}/videos/{v[0]}.mp4" for name, v in bots.items()}
        if not args.no_montage:
            path = review_dir / f"ep{ep}_inst{inst}.png"
            if montage(rec, a, b, path):
                rec["montage"] = str(path.relative_to(args.root))
        records.append(rec)
        print(f"episode {ep} instance {inst}: clean={rec['clean']} {rec['reasons']}")

    out = args.out or (args.root / "quality_manifest_shared_arena.json")
    out.write_text(json.dumps(records, indent=1))
    print(f"\nwrote {len(records)} episode records to {out}\n")

    n = len(records)
    clean = [r for r in records if r.get("clean")]
    print(f"episodes {n}, scene_check_passed {sum(bool(r.get('scene_check_passed')) for r in records)}, clean {len(clean)}, failure rate {(n - len(clean)) / max(n, 1):.0%}")
    why = collections.Counter(" ".join(w for w in x.split() if not any(c.isdigit() for c in w)) for r in records for x in r["reasons"])
    for key, cnt in why.most_common():
        print(f"  not clean: {cnt:3d} x {key}")
    print(f"\n{'ep/inst':8s} {'wallH':>5s} {'win':>7s} {'sweep plan A/B':>15s} {'yaw range A/B':>14s} {'omega A/B':>12s} {'rev A/B':>7s} {'pitch rng A/B':>13s} "
          f"{'drift':>9s} {'walls':>11s} {'ang0':>5s} {'IoU0':>5s} {'IoU med':>7s} {'ovl':>4s}")
    for r in records:
        if not r.get("recordings"):
            print(f"{r['episode'][-3:]}/{r['instance'][-1]:4s} scene failed after {r.get('scene_search_attempts')} attempts")
            continue
        A, B = list(r["recordings"].values())
        so = r.get("start_overlap") or {}
        sp = r.get("start_overlap_planned") or {}
        ovl = sum(len(rc["overlay"]["toast_frames"]) + len(rc["overlay"]["chat_text_frames"]) for rc in (A, B))
        f = lambda rc, k, d=0: rc.get(k, d) if rc.get(k) is not None else d  # noqa: E731
        print(f"{r['episode'][-3:]}/{r['instance'][-1]:4s} {str(r.get('wall_height')):>5s} {str(A.get('window_frames')) + '/' + str(B.get('window_frames')):>7s} "
              f"{(A.get('plan') or {}).get('sweep_deg', 0):6.0f}/{(B.get('plan') or {}).get('sweep_deg', 0):6.0f} "
              f"{f(A, 'yaw_range_deg'):6.0f}/{f(B, 'yaw_range_deg'):6.0f} {f(A, 'omega_measured_deg_per_sec'):5.0f}/{f(B, 'omega_measured_deg_per_sec'):5.0f} "
              f"{f(A, 'yaw_reversals'):3d}/{f(B, 'yaw_reversals'):3d} {f(A, 'pitch_max_deg') - f(A, 'pitch_min_deg'):5.0f}/{f(B, 'pitch_max_deg') - f(B, 'pitch_min_deg'):5.0f} "
              f"{f(A, 'pos_drift'):4.2f}/{f(B, 'pos_drift'):4.2f} {str(sp.get('wall_a'))[:5] + '/' + str(sp.get('wall_b'))[:5]:>11s} "
              f"{so.get('view_angle_deg', float('nan')):5.0f} {so.get('iou', float('nan')):5.2f} {(r.get('overlap') or {}).get('iou_median', float('nan')):7.2f} {ovl:4d}")
    if clean:
        yr = [rc["yaw_range_deg"] for r in clean for rc in r["recordings"].values()]
        om = [rc["omega_measured_deg_per_sec"] for r in clean for rc in r["recordings"].values()]
        s0 = [r["start_overlap"]["iou"] for r in clean]
        a0 = [r["start_overlap"]["view_angle_deg"] for r in clean]
        ious = np.concatenate([[p["iou"] for p in r["pairs"]] for r in clean])
        print(f"\nclean recordings {len(yr)}: yaw range deg min {min(yr):.0f} med {np.median(yr):.0f} max {max(yr):.0f}; "
              f"measured omega deg/s min {min(om):.0f} med {np.median(om):.0f} max {max(om):.0f}")
        print(f"start-view overlap (clean episodes): IoU min {min(s0):.2f} med {np.median(s0):.2f} max {max(s0):.2f}; "
              f"view angle deg min {min(a0):.0f} med {np.median(a0):.0f} max {max(a0):.0f}; "
              f"same/adjacent wall planned: {sum(bool((r.get('start_overlap_planned') or {}).get('same_or_adjacent_wall')) for r in clean)}/{len(clean)}")
        print(f"per-frame overlap over {len(ious)} pairs: IoU p5 {pct(ious, 5):.2f} med {pct(ious, 50):.2f} p95 {pct(ious, 95):.2f}; "
              f"share >= 0.3: {(ious >= 0.3).mean():.0%}, zero: {(ious == 0).mean():.0%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

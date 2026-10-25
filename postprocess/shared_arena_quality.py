#!/usr/bin/env python3
"""Per-episode quality manifest, review montages and the SIFT quick check
for the sharedArenaEval dataset (two bots sweeping their yaw in place inside
a built courtyard or a generated village).

Input: a flattened dataset root holding sharedArenaEval/{trajectories,
episode_info,videos}. The two recordings of an episode are paired by episode
number and instance; frames are paired by renderTime inside the motion
window (see co_observe_quality.pair_frames).

Per recording: the position must not change inside the window, the yaw
range / travel / reversals and the measured angular speed, the pitch range.
The HUD overlay heuristics of co_observe_quality are run but informational
only (they fire on light / yellow blocks).

Per episode: the measured |baseline . forward| of both bots at the first
window frame (GT poses), the start-view overlap (fraction of scene sample
points inside both frusta / inside either; courtyard: inner wall faces and
floor tops; village: exported solid blocks within 12 blocks of the bots, no
occlusion handling) and the per-frame view angle.

SIFT quick check (--sift, default on): grayscale SIFT on the first window
frame of each recording with the HUD masked (hotbar band, hand corner,
crosshair), ratio-test matches, RANSAC fundamental matrix; the inlier count
for the same-episode pair A0/B0 and for every cross-episode pair A0(i)/B0(j),
reported per scene type.

Output: <root>/quality_manifest_shared_arena.json, <root>/sift_quick_check.json,
a summary on stdout and <root>/shared_arena_review/<ep>_<inst>.png.
"""
from __future__ import annotations

import argparse
import collections
import itertools
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
MAX_BASELINE_FORWARD_DOT = 0.7
VILLAGE_POINT_RADIUS = 12.0
DEG = 180.0 / math.pi

STEM_RE = re.compile(r"^(?P<prefix>.+?)_(?P<ep>\d{6})_(?P<bot>[A-Za-z]+)_instance_(?P<inst>\d{3})$")


# ---------------------------------------------------------------------------
# Scene sample points
# ---------------------------------------------------------------------------

def courtyard_points(info: dict) -> np.ndarray:
    """Inner face centres of wall/protrusion blocks facing the yard, floor
    tops."""
    c = info["courtyard"]
    cx, cy, cz = info["centre"]
    cxi, czi = math.floor(cx), math.floor(cz)
    off = c["wall_offset"]
    pts = []
    for b in c["blocks"]:
        x, y, z = b["x"], b["y"], b["z"]
        if b.get("part") == "floor":
            pts.append((x + 0.5, y + 1.0, z + 0.5))
            continue
        dx, dz = x - cxi, z - czi
        if dx >= off:
            pts.append((x, y + 0.5, z + 0.5))
        elif dx <= -off:
            pts.append((x + 1.0, y + 0.5, z + 0.5))
        if dz >= off:
            pts.append((x + 0.5, y + 0.5, z))
        elif dz <= -off:
            pts.append((x + 0.5, y + 0.5, z + 1.0))
        if abs(dx) < off and abs(dz) < off:  # protrusion: all four side centres
            pts += [(x, y + 0.5, z + 0.5), (x + 1, y + 0.5, z + 0.5), (x + 0.5, y + 0.5, z), (x + 0.5, y + 0.5, z + 1)]
    return np.array(pts, dtype=float)


def village_points(info: dict) -> np.ndarray:
    """Centres of exported non-air blocks within VILLAGE_POINT_RADIUS of either
    bot (above the lower bot's feet)."""
    v = info["village"]
    bots = [np.array(b["pos"]) for b in info["bots"].values()]
    y0 = min(b[1] for b in bots)
    pts = []
    for e in v["blocks"]:
        x, y, z = e[0], e[1], e[2]
        if y < y0 - 1:
            continue
        p = (x + 0.5, y + 0.5, z + 0.5)
        if any(math.hypot(p[0] - b[0], p[2] - b[2]) <= VILLAGE_POINT_RADIUS for b in bots):
            pts.append(p)
    return np.array(pts, dtype=float) if pts else np.zeros((0, 3))


def scene_points(info: dict) -> np.ndarray:
    if info.get("scene_type") == "village":
        return village_points(info)
    if info.get("courtyard"):
        return courtyard_points(info)
    return np.zeros((0, 3))


# ---------------------------------------------------------------------------
# Per-recording sweep statistics
# ---------------------------------------------------------------------------

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
    out.update({
        "yaw_range_deg": float(yaw.max() - yaw.min()), "yaw_travel_deg": float(np.abs(d).sum()),
        "yaw_reversals": int(np.sum(signs[1:] != signs[:-1])) if len(signs) > 1 else 0,
        "omega_measured_deg_per_sec": float(np.median(np.abs(moving)) * 20) if len(moving) else 0.0,
        "pitch_min_deg": float(pitch.min()), "pitch_max_deg": float(pitch.max()),
        "keys_pressed": sorted({k for f in seg for k in ("forward", "back", "left", "right", "jump", "sneak", "sprint") if f["action"].get(k)}),
    })
    return out


def baseline_dots(pa: dict, pb: dict) -> dict:
    bx, bz = pb["x"] - pa["x"], pb["z"] - pa["z"]
    n = math.hypot(bx, bz) or 1.0
    fa, fb = forward(pa["yaw"], pa["pitch"]), forward(pb["yaw"], pb["pitch"])
    return {"a": abs((bx * fa[0] + bz * fa[2]) / n), "b": abs((bx * fb[0] + bz * fb[2]) / n), "distance": n}


# ---------------------------------------------------------------------------
# Episode analysis
# ---------------------------------------------------------------------------

def analyze_episode(a, b) -> dict:
    (base_a, info_a, fr_a, vid_a), (base_b, info_b, fr_b, vid_b) = a, b
    rec = {
        "type": TYPE,
        "scene_type": info_a.get("scene_type"),
        "bots": [info_a["bot_name"], info_b["bot_name"]],
        "primary": info_a.get("primary"),
        "world_type": info_a.get("world_type"),
        "scene_check_passed": bool(info_a.get("scene_check_passed")) and bool(info_b.get("scene_check_passed")),
        "centre": info_a.get("centre"),
        "start_overlap_planned": info_a.get("start_overlap"),
        "scene_search_attempts": info_a.get("scene_search_attempts") or info_b.get("scene_search_attempts"),
        "encountered_error": bool(info_a.get("encountered_error") or info_b.get("encountered_error")),
        "reasons": [],
    }
    if info_a.get("courtyard"):
        c = info_a["courtyard"]
        rec["courtyard"] = {"palette": c.get("palette"), "modules": c.get("modules"), "num_blocks": c.get("num_blocks"), "max_height": c.get("max_height")}
    if info_a.get("village"):
        v = info_a["village"]
        rec["village"] = {"located": v.get("located"), "search": v.get("search"), "export_blocks": len(v.get("blocks", [])),
                          "ray_coverage": {n: bb.get("ray_coverage") for n, bb in (info_a.get("bots") or {}).items()},
                          "built_fraction": {n: bb.get("built_fraction") for n, bb in (info_a.get("bots") or {}).items()},
                          "sky_light": {n: bb.get("sky_light") for n, bb in (info_a.get("bots") or {}).items()},
                          "world_edits_for_placement": info_a.get("world_edits_for_placement", [])}
    if not rec["scene_check_passed"]:
        rec["reasons"].append("scene setup failed")
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
    pairs = pair_frames(fr_a, fr_b, wa, wb) if wa and wb else []
    rec["num_pairs"] = len(pairs)
    if wa and wb and len(pairs) < MIN_WINDOW_FRAMES:
        rec["reasons"].append(f"only {len(pairs)} paired frames")
    pts = scene_points(info_a)
    per = []
    for ka, kb in pairs:
        pa, pb = fr_a[ka], fr_b[kb]
        entry = {"ka": ka, "kb": kb, "dt_ms": pa["renderTime"] - pb["renderTime"],
                 "view_angle_deg": angle_between(forward(pa["yaw"], pa["pitch"]), forward(pb["yaw"], pb["pitch"])),
                 "yaw_a_deg": pa["yaw"] * DEG, "yaw_b_deg": pb["yaw"] * DEG}
        if len(pts):
            sa = visible_many(pa, pts, va.width, va.height)
            sb = visible_many(pb, pts, vb.width, vb.height)
            union = int((sa | sb).sum())
            entry["both"] = int((sa & sb).sum())
            entry["iou"] = float((sa & sb).sum() / union) if union else 0.0
        per.append(entry)
    rec["pairs"] = per
    if per:
        pa0, pb0 = fr_a[per[0]["ka"]], fr_b[per[0]["kb"]]
        rec["baseline_forward_dot_measured"] = baseline_dots(pa0, pb0)
        if max(rec["baseline_forward_dot_measured"]["a"], rec["baseline_forward_dot_measured"]["b"]) > MAX_BASELINE_FORWARD_DOT + 0.02:
            rec["reasons"].append("|baseline.forward| > 0.7 at frame 0")
        rec["start_overlap"] = {"view_angle_deg": per[0]["view_angle_deg"], "iou": per[0].get("iou"), "points_both": per[0].get("both"),
                                "points_total": int(len(pts))}
        if len(pts):
            ious = np.array([p["iou"] for p in per])
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
        txt = (f"{rec.get('scene_type')} | {info_a['bot_name']} f{p['ka']} yaw {p['yaw_a_deg']:.0f} | {info_b['bot_name']} f{p['kb']} yaw {p['yaw_b_deg']:.0f} | "
               f"view angle {p['view_angle_deg']:.0f} deg" + (f"  overlap IoU {p['iou']:.2f}" if "iou" in p else ""))
        cv2.putText(row, txt, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(row, txt, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        rows.append(row)
    va.close()
    vb.close()
    if not rows:
        return False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(out_path), np.concatenate(rows, axis=0)))


# ---------------------------------------------------------------------------
# SIFT quick check
# ---------------------------------------------------------------------------

def hud_mask(h: int, w: int) -> np.ndarray:
    """255 where features may be used: the hotbar band, the hand corner and
    the crosshair are excluded."""
    m = np.full((h, w), 255, np.uint8)
    m[int(h * 0.82):, :] = 0                          # hotbar / health bar band
    m[int(h * 0.62):, int(w * 0.68):] = 0              # first-person hand
    cy, cx = h // 2, w // 2
    m[cy - 25: cy + 25, cx - 25: cx + 25] = 0          # crosshair
    return m


def sift_features(img, sift):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kp, des = sift.detectAndCompute(g, hud_mask(*g.shape))
    return kp, des


def inliers(kp1, d1, kp2, d2, matcher) -> dict:
    if d1 is None or d2 is None or len(kp1) < 8 or len(kp2) < 8:
        return {"matches": 0, "inliers": 0}
    raw = matcher.knnMatch(d1, d2, k=2)
    good = [m for m, n in (p for p in raw if len(p) == 2) if m.distance < 0.75 * n.distance]
    if len(good) < 15:
        return {"matches": len(good), "inliers": 0}
    p1 = np.float32([kp1[m.queryIdx].pt for m in good])
    p2 = np.float32([kp2[m.trainIdx].pt for m in good])
    F, mask = cv2.findFundamentalMat(p1, p2, cv2.FM_RANSAC, 1.0, 0.999)
    # On failure OpenCV returns F None and an uninitialised mask.
    if F is None or mask is None or F.shape[0] < 3:
        return {"matches": len(good), "inliers": 0}
    return {"matches": len(good), "inliers": int((mask.ravel() == 1).sum())}


def sift_quick_check(records: list, loaded: dict) -> dict:
    sift = cv2.SIFT_create()
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    feats = {}  # (inst, ep) -> {"A": (kp, des), "B": (kp, des), "scene": type}
    for r in records:
        if not r.get("recordings") or not r.get("pairs"):
            continue
        key = (r["instance"], r["episode"])
        bots = loaded[key]
        names = sorted(bots)
        p0 = r["pairs"][0]
        entry = {"scene": r.get("scene_type")}
        for tag, name, k in (("A", names[0], p0["ka"]), ("B", names[1], p0["kb"])):
            v = Video(bots[name][3])
            img = v.frame(k)
            v.close()
            if img is None:
                continue
            kp, des = sift_features(img, sift)
            entry[tag] = (kp, des, len(kp))
        if "A" in entry and "B" in entry:
            feats[key] = entry
    same = []
    cross = []
    for key, e in feats.items():
        res = inliers(e["A"][0], e["A"][1], e["B"][0], e["B"][1], matcher)
        same.append({"episode": key[1], "instance": key[0], "scene": e["scene"], "keypoints_a": e["A"][2], "keypoints_b": e["B"][2], **res})
    for ka, kb in itertools.permutations(feats.keys(), 2):
        ea, eb = feats[ka], feats[kb]
        res = inliers(ea["A"][0], ea["A"][1], eb["B"][0], eb["B"][1], matcher)
        cross.append({"a_episode": ka[1], "a_instance": ka[0], "b_episode": kb[1], "b_instance": kb[0],
                      "scene_a": ea["scene"], "scene_b": eb["scene"], **res})
    return {"mask": "hotbar band (bottom 18%), hand corner (x>68%, y>62%), crosshair 50x50", "same_episode": same, "cross_episode": cross}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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


def summarize(vals):
    vals = sorted(vals)
    if not vals:
        return "n/a"
    return f"n {len(vals)} min {vals[0]} med {vals[len(vals) // 2]} max {vals[-1]}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--no-montage", action="store_true")
    ap.add_argument("--no-sift", action="store_true")
    args = ap.parse_args()

    loaded = load(args.root)
    records = []
    review_dir = args.root / "shared_arena_review"
    for (inst, ep), bots in sorted(loaded.items()):
        names = sorted(bots)
        if len(names) < 2:
            records.append({"type": TYPE, "episode": ep, "instance": inst, "bots": names, "clean": False,
                            "scene_type": bots[names[0]][1].get("scene_type"),
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
        print(f"episode {ep} instance {inst}: scene={rec.get('scene_type')} clean={rec['clean']} {rec['reasons']}")

    out = args.out or (args.root / "quality_manifest_shared_arena.json")
    out.write_text(json.dumps(records, indent=1))
    print(f"\nwrote {len(records)} episode records to {out}\n")

    n = len(records)
    clean = [r for r in records if r.get("clean")]
    print(f"episodes {n}, scene_check_passed {sum(bool(r.get('scene_check_passed')) for r in records)}, clean {len(clean)}, failure rate {(n - len(clean)) / max(n, 1):.0%}")
    for key, cnt in collections.Counter(" ".join(w for w in x.split() if not any(c.isdigit() for c in w)) for r in records for x in r["reasons"]).most_common():
        print(f"  not clean: {cnt:3d} x {key}")

    print(f"\n{'ep/inst':8s} {'scene':9s} {'win':>7s} {'yaw range A/B':>14s} {'omega A/B':>10s} {'drift':>9s} {'dist':>4s} {'|b.f| A/B plan':>14s} {'|b.f| A/B meas':>14s} {'ang0':>5s} {'IoU0':>5s} {'IoU med':>7s} {'cov A/B':>9s}")
    for r in records:
        if not r.get("recordings"):
            print(f"{r['episode'][-3:]}/{r['instance'][-1]:4s} {str(r.get('scene_type')):9s} scene setup failed after {r.get('scene_search_attempts')} attempts")
            continue
        A, B = list(r["recordings"].values())
        so = r.get("start_overlap") or {}
        sp = r.get("start_overlap_planned") or {}
        pd = sp.get("baseline_forward_dot") or {}
        md = r.get("baseline_forward_dot_measured") or {}
        cov = (r.get("village") or {}).get("ray_coverage") or {}
        covs = "/".join(f"{v:.2f}" for v in cov.values()) if cov else "-"
        f = lambda rc, k, d=0: rc.get(k, d) if rc.get(k) is not None else d  # noqa: E731
        print(f"{r['episode'][-3:]}/{r['instance'][-1]:4s} {str(r.get('scene_type')):9s} {str(A.get('window_frames')) + '/' + str(B.get('window_frames')):>7s} "
              f"{f(A, 'yaw_range_deg'):6.0f}/{f(B, 'yaw_range_deg'):6.0f} {f(A, 'omega_measured_deg_per_sec'):4.0f}/{f(B, 'omega_measured_deg_per_sec'):4.0f} "
              f"{f(A, 'pos_drift'):4.2f}/{f(B, 'pos_drift'):4.2f} {md.get('distance', float('nan')):4.1f} "
              f"{pd.get('a', float('nan')):6.2f}/{pd.get('b', float('nan')):6.2f} {md.get('a', float('nan')):6.2f}/{md.get('b', float('nan')):6.2f} "
              f"{so.get('view_angle_deg', float('nan')):5.0f} {(so.get('iou') if so.get('iou') is not None else float('nan')):5.2f} "
              f"{(r.get('overlap') or {}).get('iou_median', float('nan')):7.2f} {covs:>9s}")

    for scene in ("courtyard", "village"):
        sub = [r for r in records if r.get("scene_type") == scene]
        cl = [r for r in sub if r.get("clean")]
        if not sub:
            continue
        print(f"\n[{scene}] episodes {len(sub)}, clean {len(cl)}, scene setup failed {sum(not r.get('scene_check_passed') for r in sub)}")
        if cl:
            dots = [max(r["baseline_forward_dot_measured"]["a"], r["baseline_forward_dot_measured"]["b"]) for r in cl]
            print(f"  max |baseline.forward| per episode (measured, frame 0): " + " ".join(f"{d:.2f}" for d in dots))
            yr = [rc["yaw_range_deg"] for r in cl for rc in r["recordings"].values()]
            print(f"  yaw range deg: min {min(yr):.0f} med {np.median(yr):.0f} max {max(yr):.0f}")
            a0 = [r["start_overlap"]["view_angle_deg"] for r in cl]
            print(f"  start view angle deg: min {min(a0):.0f} med {np.median(a0):.0f} max {max(a0):.0f}")
            if all(r["start_overlap"].get("iou") is not None for r in cl):
                s0 = [r["start_overlap"]["iou"] for r in cl]
                print(f"  start-view overlap IoU: min {min(s0):.2f} med {np.median(s0):.2f} max {max(s0):.2f}")
        if scene == "village":
            covs = [v for r in sub for v in ((r.get("village") or {}).get("ray_coverage") or {}).values() if v is not None]
            if covs:
                print(f"  ray coverage of chosen spots: min {min(covs):.2f} med {np.median(covs):.2f} max {max(covs):.2f}")
            for r in sub:
                v = r.get("village") or {}
                s = v.get("search") or {}
                bf = "/".join(f"{x:.2f}" for x in (v.get("built_fraction") or {}).values() if x is not None)
                sl = "/".join(str(x) for x in (v.get("sky_light") or {}).values())
                print(f"  {r['episode'][-3:]}/{r['instance'][-1]}: spots passing {s.get('spots_found')} of {s.get('columns_checked')} standable columns "
                      f"(rejected {s.get('rejected')}), attempts {r.get('scene_search_attempts')}, built fraction A/B {bf}, sky light {sl}, "
                      f"exported blocks {v.get('export_blocks')}, edits {v.get('world_edits_for_placement')}")

    if not args.no_sift:
        qc = sift_quick_check(records, loaded)
        (args.root / "sift_quick_check.json").write_text(json.dumps(qc, indent=1))
        print(f"\nSIFT quick check (grayscale SIFT, ratio 0.75, RANSAC F 1.0 px; HUD masked: {qc['mask']})")
        for scene in ("courtyard", "village"):
            same = [e["inliers"] for e in qc["same_episode"] if e["scene"] == scene]
            cross_same_scene = [e["inliers"] for e in qc["cross_episode"] if e["scene_a"] == scene and e["scene_b"] == scene]
            kps = [e["keypoints_a"] for e in qc["same_episode"] if e["scene"] == scene] + [e["keypoints_b"] for e in qc["same_episode"] if e["scene"] == scene]
            print(f"  [{scene}] keypoints per frame: {summarize(kps)}")
            print(f"  [{scene}] same-episode A0/B0 inliers:            {summarize(same)}  values {sorted(same)}")
            print(f"  [{scene}] cross-episode A0(i)/B0(j) inliers:     {summarize(cross_same_scene)}")
        mixed = [e["inliers"] for e in qc["cross_episode"] if e["scene_a"] != e["scene_b"]]
        print(f"  [mixed scenes] cross-episode inliers:              {summarize(mixed)}")
        cross_all = sorted(e["inliers"] for e in qc["cross_episode"])
        if cross_all:
            print(f"  cross-episode inliers percentiles: p50 {pct(cross_all, 50):.0f} p90 {pct(cross_all, 90):.0f} p100 {cross_all[-1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

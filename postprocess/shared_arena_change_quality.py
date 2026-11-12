#!/usr/bin/env python3
"""Quality manifest and before / after review images for sharedArenaChangeEval
(sharedArenaEval courtyards where one wall module changes in the middle of
the window while both bots hold their look on it).

Input: a flattened dataset root holding sharedArenaChangeEval/{trajectories,
episode_info,videos}. The base checks are those of shared_arena_quality
(static position, sweep, start overlap). On top, per recording:

  * change.frame (trajectory frame at which the bot's client received the
    block change) must lie inside the hold;
  * the fraction of the module's inner-face block centres inside the frame
    at that pose (GT projection) must be >= MIN_VISIBLE_FRACTION;
  * video_change_frame: the frame of the video where the picture inside the
    projected module box jumps while the rest of the picture is still (the
    camera is static during the hold). This is the pixel-exact frame; it can
    trail change.frame by a few frames. A jump is a frame-to-frame difference
    inside the box >= JUMP_MIN grey levels and >= JUMP_RATIO times the
    difference outside it.

Per episode a 2x2 image (rows: before / after the change, columns: the two
bots; the projected box of the changed module outlined) is written to
<root>/shared_arena_change_review/ep<ep>_inst<inst>_<type>.png.

Output: <root>/quality_manifest_shared_arena_change.json and a summary.
"""
from __future__ import annotations

import argparse
import collections
import itertools
import json
from pathlib import Path

import cv2
import numpy as np

import shared_arena_quality as saq
from co_observe_quality import Video, project_many

TYPE = "sharedArenaChangeEval"
MIN_VISIBLE_FRACTION = 0.7
SEARCH_BEFORE = 6     # frames before change.frame searched for the pixel change
SEARCH_AFTER = 14
STILL_MAX = 3.0       # mean abs grey difference outside the box for a "still" frame pair (a sweep gives >> 3;
                      # the change itself spills ~1 outside the box: light / shading of the neighbouring blocks)
JUMP_MIN = 1.0        # mean abs grey difference inside the box at the change (the box is mostly sky / floor
JUMP_RATIO = 5.0      # around a module seen from afar), and at least this many times the difference outside
BEFORE_OFFSET = 3     # review frames: change - 3, change + 5
AFTER_OFFSET = 5


def box_corners(ch: dict) -> np.ndarray:
    lo, hi = np.array(ch["box_min"], float), np.array(ch["box_max"], float) + 1.0
    return np.array(list(itertools.product(*zip(lo, hi))))


def face_points(ch: dict) -> np.ndarray:
    """Inner-face centres of the module's wall cells (3 wide, module_height tall)."""
    lo, hi = np.array(ch["box_min"], float), np.array(ch["box_max"], float) + 1.0
    fc = np.array(ch["face_centre"], float)
    along = 0 if ch["side"] in ("south", "north") else 2
    pts = []
    for a in np.arange(lo[along] + 0.5, hi[along]):
        for v in range(int(ch["module_height"])):
            p = fc.copy()
            p[along] = a
            p[1] = lo[1] + v + 0.5
            pts.append(p)
    return np.array(pts)


def box_mask(pose: dict, ch: dict, w: int, h: int):
    px, py, ok = project_many(pose, box_corners(ch), w, h)
    if not ok.all():
        return None, None
    hull = cv2.convexHull(np.stack([px, py], 1).astype(np.float32)).astype(np.int32)
    mask = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(mask, hull, 1)
    return mask.astype(bool), hull


def analyze_change(info: dict, frames: list, vid: Video) -> dict:
    ch = info.get("change") or {}
    out = {"bot": info["bot_name"], "change_frame": ch.get("frame"), "hold_frames": [ch.get("hold_start_frame"), ch.get("hold_end_frame")],
           "seen_tick": ch.get("seen_tick"), "client_cells_not_as_planned": ch.get("client_cells_not_as_planned"), "reasons": []}
    cf = ch.get("frame")
    if cf is None:
        out["reasons"].append("change not seen by the client")
        return out
    h0, h1 = ch.get("hold_start_frame"), ch.get("hold_end_frame")
    if h0 is None or h1 is None or not (h0 < cf < h1):
        out["reasons"].append(f"change frame {cf} outside the hold {h0}..{h1}")
    else:
        out["hold_before_frames"], out["hold_after_frames"] = cf - h0, h1 - cf
        seg = frames[h0: h1]
        out["hold_yaw_range_deg"] = float((max(f["yaw"] for f in seg) - min(f["yaw"] for f in seg)) * saq.DEG)
    if ch.get("client_cells_not_as_planned"):
        out["reasons"].append(f"{ch['client_cells_not_as_planned']} cells of the box not as planned after the change")
    pose = frames[min(cf, len(frames) - 1)]
    pts = face_points(ch)
    px, py, ok = project_many(pose, pts, vid.width, vid.height)
    out["visible_fraction"] = float((ok & (px >= 0) & (px < vid.width) & (py >= 0) & (py < vid.height)).mean())
    if out["visible_fraction"] < MIN_VISIBLE_FRACTION:
        out["reasons"].append(f"only {out['visible_fraction']:.2f} of the module face in frame")
    mask, hull = box_mask(pose, ch, vid.width, vid.height)
    out["_hull"] = hull
    if mask is None or not mask.any():
        out["reasons"].append("module box not in front of the camera")
        return out
    out["box_pixel_fraction"] = float(mask.mean())
    best = None
    prev = None
    series = []
    for k in range(max(1, cf - SEARCH_BEFORE - 1), min(vid.n, cf + SEARCH_AFTER + 1)):
        img = vid.frame(k)
        if img is None:
            prev = None
            continue
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if prev is not None:
            d = np.abs(g - prev)
            inside, outside = float(d[mask].mean()), float(d[~mask].mean())
            series.append([k, round(inside, 2), round(outside, 2)])
            if outside <= STILL_MAX and inside >= max(JUMP_MIN, JUMP_RATIO * outside) and (best is None or inside > best[1]):
                best = (k, inside)
        prev = g
    out["diff_series"] = series
    if best is None:
        out["reasons"].append("no pixel change found in the module box during the hold")
        return out
    out["video_change_frame"] = best[0]
    out["video_change_diff"] = round(best[1], 2)
    out["video_lag_frames"] = best[0] - cf
    return out


def review_image(rec: dict, a, b, out_path: Path) -> bool:
    rows = []
    for label, off in (("before", -BEFORE_OFFSET), ("after", AFTER_OFFSET)):
        cells = []
        for (base, info, frames, vid_path) in (a, b):
            m = rec["change_recordings"][info["bot_name"]]
            k0 = m.get("video_change_frame", m.get("change_frame"))
            if k0 is None:
                return False
            vid = Video(vid_path)
            img = vid.frame(k0 + off)
            vid.close()
            if img is None:
                return False
            img = img.copy()
            if m.get("_hull") is not None:
                cv2.polylines(img, [m["_hull"]], True, (0, 255, 255), 2, cv2.LINE_AA)
            txt = f"{label}: {info['bot_name']} frame {k0 + off} (change at {k0})"
            cv2.putText(img, txt, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(img, txt, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
            cells.append(img)
        rows.append(np.concatenate(cells, axis=1))
    img = np.concatenate(rows, axis=0)
    c = rec["change"]
    what = c.get("module_after") or c.get("palette_after") or c.get("plain_block")
    title = (f"ep {rec['episode']} inst {rec['instance']} | {c['type']} / {c['subtype']} | {c['side']} wall slot {c['index']}: {c['module_before']} "
             f"({c['palette_before']}) -> {what} | window tick {c['tick']} | box {c['box_min']}..{c['box_max']}")
    bar = np.full((44, img.shape[1], 3), 32, np.uint8)
    cv2.putText(bar, title, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(out_path), np.concatenate([bar, img], axis=0)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    saq.TYPE = TYPE
    loaded = saq.load(args.root)
    review_dir = args.root / "shared_arena_change_review"
    records = []
    for (inst, ep), bots in sorted(loaded.items()):
        names = sorted(bots)
        planned = bots[names[0]][1].get("change_planned") or {}
        if len(names) < 2:
            records.append({"type": TYPE, "episode": ep, "instance": inst, "bots": names, "clean": False, "change_planned": planned,
                            "reasons": ["second recording missing"]})
            continue
        a, b = bots[names[0]], bots[names[1]]
        rec = saq.analyze_episode(a, b)
        rec.update({"type": TYPE, "episode": ep, "instance": inst, "change_planned": planned,
                    "recording_files": {name: f"{TYPE}/videos/{v[0]}.mp4" for name, v in bots.items()}})
        rec.pop("pairs", None)
        if rec.get("scene_check_passed"):
            primary = next((v[1] for v in bots.values() if (v[1].get("change") or {}).get("executed_by")), a[1])
            ch = primary.get("change") or {}
            rec["change"] = {k: ch.get(k) for k in ("type", "subtype", "tick", "hold_start_tick", "hold_end_tick", "side", "index", "module_before",
                                                    "module_after", "palette_before", "palette_after", "plain_block", "module_height", "box_min",
                                                    "box_max", "num_cells_changed", "executed_by", "response", "ok")}
            if not ch.get("ok"):
                rec["reasons"].append(f"clone not confirmed: {ch.get('response')}")
            rec["change_recordings"] = {}
            for (base, info, frames, vid_path) in (a, b):
                vid = Video(vid_path)
                m = analyze_change(info, frames, vid)
                vid.close()
                rec["change_recordings"][info["bot_name"]] = m
                rec["reasons"] += [f"{info['bot_name']}: {r}" for r in m.pop("reasons")]
            path = review_dir / f"ep{ep}_inst{inst}_{ch.get('type')}.png"
            if review_image(rec, a, b, path):
                rec["review_image"] = str(path.relative_to(args.root))
            for m in rec["change_recordings"].values():
                m.pop("_hull", None)
            rec["clean"] = not rec["reasons"]
        records.append(rec)
        print(f"episode {ep} instance {inst}: {planned.get('kind')}/{planned.get('subtype')} clean={rec['clean']} {rec['reasons']}")

    out = args.out or (args.root / "quality_manifest_shared_arena_change.json")
    out.write_text(json.dumps(records, indent=1))
    clean = [r for r in records if r.get("clean")]
    print(f"\nwrote {len(records)} episode records to {out}")
    print(f"episodes {len(records)}, clean {len(clean)}; clean by type: {dict(collections.Counter(r['change']['type'] + '/' + r['change']['subtype'] for r in clean))}")
    print(f"\n{'ep/inst':8s} {'type':20s} {'module -> after':44s} {'tick':>4s} {'seen':>9s} {'frame A/B':>9s} {'video A/B':>9s} {'visible':>9s} {'box px %':>9s} {'cells':>5s}")
    for r in records:
        c, cr = r.get("change"), r.get("change_recordings")
        if not c or not cr:
            print(f"{r['episode'][-3:]}/{r['instance'][-1]:4s} {str((r.get('change_planned') or {}).get('kind')):20s} not recorded: {r['reasons']}")
            continue
        A, B = list(cr.values())
        what = c.get("module_after") or c.get("palette_after") or c.get("plain_block")
        j = lambda k, fmt="{}": "/".join("-" if m.get(k) is None else fmt.format(m[k]) for m in (A, B))  # noqa: E731
        print(f"{r['episode'][-3:]}/{r['instance'][-1]:4s} {c['type'] + '/' + c['subtype']:20s} {c['module_before'] + ' -> ' + str(what):44s} {c['tick']:4d} {j('seen_tick'):>9s} "
              f"{j('change_frame'):>9s} {j('video_change_frame'):>9s} {j('visible_fraction', '{:.2f}'):>9s} "
              f"{'/'.join('-' if m.get('box_pixel_fraction') is None else format(100 * m['box_pixel_fraction'], '.1f') for m in (A, B)):>9s} {c['num_cells_changed']:5d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

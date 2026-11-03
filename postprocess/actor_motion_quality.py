#!/usr/bin/env python3
"""Per-episode quality manifest and review montages for the actorMotionEval
dataset (one atomic body action of the actor A per episode, forward / back /
left / right / jump / sneak, watched by the static observer B).

For every episode it checks

  window           105-129 frames (window_start_frame .. window_end_frame)
  observer_static  B's pose is unchanged and no key is pressed over the
                   whole window
  actor_static     A's pose is unchanged before the action (>= MIN_PRE_FRAMES
                   frames) and after it (>= MIN_POST_FRAMES frames), and A's
                   yaw / pitch never change over the window (no camera action)
  action           the only key pressed during the action frames is the
                   expected one, and the measured motion matches: >=
                   MIN_WALK_BLOCKS along the key's direction in A's frame
                   (and nothing across it) for a walk, a rise >= MIN_JUMP_RISE
                   and no horizontal motion for a jump, sneak held with no
                   motion for a sneak
  in_frame         A's body box (0.6 x 1.8, 1.5 while sneaking), placed with
                   A's pose at the same wall-clock time, projects fully inside
                   B's frame on every window frame (GT pose, vertical FOV 70
                   deg); the projected pixel height of A is reported per
                   episode (start / min / max / end)
  los_clear        line of sight from B's eye to A's body before and after
                   the action (controller, through the world)
  pixel_change     B's video changes inside A's projected box between the
                   frame before the action and the frame after (mid-jump /
                   mid-sneak for those), far more than the rest of the frame
  overlay          toast heuristic of co_observe_quality inside the window

Output: <root>/quality_manifest_actor_motion.json, a summary on stdout (with
the pixel height of A per observer distance) and
<root>/actor_motion_review/ep<ep>_inst<inst>.png (observer | actor views
before, during, after the action and at the window end, A's projected box
drawn on the observer frames).
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import statistics
from pathlib import Path

import cv2
import numpy as np

import world_mod_quality as wmq
from co_observe_quality import overlay_scan
from traj_segments import resolve_event_frames

TYPE = "actorMotionEval"
wmq.TYPE = TYPE
WINDOW_FRAMES = (105, 129)
MIN_PRE_FRAMES = 16
MIN_POST_FRAMES = 40
MIN_WALK_BLOCKS = 1.5
MAX_CROSS_BLOCKS = 0.3
MIN_JUMP_RISE = 1.0
MIN_SNEAK_FRAMES = 12
BODY_HALF = 0.3
BODY_HEIGHT = 1.8
SNEAK_HEIGHT = 1.5
PIXEL_CHANGE_MIN = 6.0
PIXEL_CHANGE_RATIO = 2.0
BOX_PAD_PX = 12
DEG = 180.0 / math.pi
MOVE_KEYS = ("forward", "back", "left", "right", "jump", "sneak", "sprint")


def window_of(info: dict):
    w0, w1 = info.get("window_start_frame"), info.get("window_end_frame")
    return (w0, w1) if w0 is not None and w1 is not None else None


def nearest_index(times: list, t: float) -> int:
    """Index of the frame whose renderTime is closest to t (times sorted)."""
    import bisect
    k = bisect.bisect_left(times, t)
    if k == 0:
        return 0
    if k >= len(times):
        return len(times) - 1
    return k if abs(times[k] - t) < abs(times[k - 1] - t) else k - 1


def body_corners(pose: dict):
    h = SNEAK_HEIGHT if pose["action"].get("sneak") else BODY_HEIGHT
    return [(pose["x"] + dx, pose["y"] + dy, pose["z"] + dz)
            for dx in (-BODY_HALF, BODY_HALF) for dy in (0.0, h) for dz in (-BODY_HALF, BODY_HALF)]


def project_body(obs_pose: dict, act_pose: dict, width: int, height: int):
    """Projected box [x0, y0, x1, y1] of A's body in B's frame, or None when a
    corner is behind the camera."""
    pts = [wmq.project(obs_pose, c, width, height) for c in body_corners(act_pose)]
    if any(p is None for p in pts):
        return None
    return [min(p[0] for p in pts), min(p[1] for p in pts), max(p[0] for p in pts), max(p[1] for p in pts)]


def box_in_frame(box, width, height):
    return box is not None and wmq.in_frame((box[0], box[1]), width, height) and wmq.in_frame((box[2], box[3]), width, height)


def key_direction(key: str, yaw: float):
    offset = {"forward": 0.0, "left": 90.0, "back": 180.0, "right": -90.0}[key]
    a = yaw + math.radians(offset)
    return (-math.sin(a), -math.cos(a))


def camera_static(frames, lo, hi):
    seg = frames[lo: hi + 1]
    if not seg:
        return {"frames": 0, "ok": False}
    ref = seg[0]
    dyaw = max(abs((((f["yaw"] - ref["yaw"]) + math.pi) % (2 * math.pi)) - math.pi) for f in seg) * DEG
    dpitch = max(abs(f["pitch"] - ref["pitch"]) for f in seg) * DEG
    cam = max(max(abs(c) for c in f["action"].get("camera", [0, 0])) for f in seg)
    return {"frames": len(seg), "dyaw_deg": dyaw, "dpitch_deg": dpitch, "camera_max": cam,
            "ok": dyaw <= wmq.POSE_EPS_DEG and dpitch <= wmq.POSE_EPS_DEG and cam <= wmq.CAMERA_EPS_RAD}


def actor_motion(info: dict, frames: list, ev: dict, win) -> dict:
    """What A actually did, from its own trajectory."""
    fs, fe = ev["frame_start"], ev["frame_end"]
    w0, w1 = win
    out = {"action": ev["type"], "reasons": []}
    ok_b, det_b = wmq.pose_static(frames, w0, fs - 1)
    ok_a, det_a = wmq.pose_static(frames, fe + 1, w1)
    out["static_before"], out["static_after"] = det_b, det_a
    if det_b["frames"] < MIN_PRE_FRAMES or not ok_b:
        out["reasons"].append(f"actor not static before the action ({det_b['frames']} frames)")
    if det_a["frames"] < MIN_POST_FRAMES or not ok_a:
        out["reasons"].append(f"actor not static after the action ({det_a['frames']} frames)")
    out["camera"] = camera_static(frames, w0, w1)
    if not out["camera"]["ok"]:
        out["reasons"].append("actor camera moved during the window")
    during = frames[fs: fe + 1]
    keys = collections.Counter(k for f in during for k in MOVE_KEYS if f["action"].get(k))
    out["keys_during"] = dict(keys)
    expected = ev["key"]
    if set(keys) - {expected}:
        out["reasons"].append(f"unexpected keys during the action: {sorted(set(keys) - {expected})}")
    if keys.get(expected, 0) == 0:
        out["reasons"].append(f"{expected} key never pressed in the action frames")
    p0, p1 = frames[max(fs - 1, 0)], frames[min(fe + 1, len(frames) - 1)]
    disp = (p1["x"] - p0["x"], p1["y"] - p0["y"], p1["z"] - p0["z"])
    fwd, left = key_direction("forward", p0["yaw"]), key_direction("left", p0["yaw"])
    along = disp[0] * fwd[0] + disp[2] * fwd[1]
    across = disp[0] * left[0] + disp[2] * left[1]
    horiz = math.hypot(disp[0], disp[2])
    rise = max(f["y"] for f in during) - p0["y"] if during else 0.0
    out.update({"displacement": disp, "forward_blocks": along, "left_blocks": across, "horizontal_blocks": horiz,
                "max_y_rise": rise, "sneak_frames": keys.get("sneak", 0), "motion_frames": fe - fs + 1})
    a = ev["type"]
    if a in ("forward", "back", "left", "right"):
        signed = {"forward": along, "back": -along, "left": across, "right": -across}[a]
        cross = across if a in ("forward", "back") else along
        if signed < MIN_WALK_BLOCKS:
            out["reasons"].append(f"moved {signed:.2f} blocks along {a}, expected >= {MIN_WALK_BLOCKS}")
        if abs(cross) > MAX_CROSS_BLOCKS:
            out["reasons"].append(f"moved {cross:.2f} blocks across the key direction")
        if abs(disp[1]) > 0.05 or rise > 0.1:
            out["reasons"].append("vertical motion during a walk")
    elif a == "jump":
        if rise < MIN_JUMP_RISE:
            out["reasons"].append(f"jump rise {rise:.2f} < {MIN_JUMP_RISE}")
        if horiz > MAX_CROSS_BLOCKS:
            out["reasons"].append(f"moved {horiz:.2f} blocks during the jump")
        if abs(disp[1]) > 0.05:
            out["reasons"].append("did not land at the start height")
    elif a == "sneak":
        if keys.get("sneak", 0) < MIN_SNEAK_FRAMES:
            out["reasons"].append(f"sneak held {keys.get('sneak', 0)} frames < {MIN_SNEAK_FRAMES}")
        if horiz > 0.05 or abs(disp[1]) > 0.01:
            out["reasons"].append("moved during the sneak")
    return out


def observer_view(obs_info: dict, obs_frames: list, act_frames: list, win, ev: dict, video: wmq.Video) -> dict:
    """A's projected body box in B's frames over the window, pixel heights,
    in-frame check, and the rendered change inside the box."""
    w0, w1 = win
    out = {"reasons": []}
    ok, det = wmq.pose_static(obs_frames, w0, w1)
    out["static"] = det
    if not ok:
        out["reasons"].append("observer not static over the window")
    act_times = [f["renderTime"] for f in act_frames]
    boxes = {}
    for k in range(w0, w1 + 1):
        ka = nearest_index(act_times, obs_frames[k]["renderTime"])
        boxes[k] = project_body(obs_frames[k], act_frames[ka], video.width, video.height)
    out["frames_out_of_frame"] = [k for k, b in boxes.items() if not box_in_frame(b, video.width, video.height)]
    if out["frames_out_of_frame"]:
        out["reasons"].append(f"actor outside the observer frame on {len(out['frames_out_of_frame'])} frames")
    heights = {k: (b[3] - b[1]) for k, b in boxes.items() if b is not None}
    if heights:
        out["px_height"] = {"start": heights[w0], "end": heights[w1], "min": min(heights.values()), "max": max(heights.values())}
        b0, b1 = boxes[w0], boxes[w1]
        out["px_centre_start"] = [(b0[0] + b0[2]) / 2, (b0[1] + b0[3]) / 2] if b0 else None
        out["px_centre_end"] = [(b1[0] + b1[2]) / 2, (b1[1] + b1[3]) / 2] if b1 else None
        out["px_width_start"] = (b0[2] - b0[0]) if b0 else None
    out["boxes"] = {str(k): [round(v, 1) for v in b] for k, b in boxes.items() if b is not None and k in (w0, ev["frame_start"], ev["frame_end"], w1)}

    # Rendered change: the union of A's boxes before and after (mid-action
    # for jump / sneak) vs the rest of the frame.
    fs, fe = ev["frame_start"], ev["frame_end"]
    k_before = max(w0, fs - 4)
    k_after = (fs + fe) // 2 if ev["type"] in ("jump", "sneak") else min(w1, fe + 4)
    a_img, b_img = video.frame(k_before), video.frame(k_after)
    ba, bb = boxes.get(k_before), boxes.get(k_after)
    if a_img is not None and b_img is not None and ba and bb:
        x0 = int(max(0, min(ba[0], bb[0]) - BOX_PAD_PX))
        y0 = int(max(0, min(ba[1], bb[1]) - BOX_PAD_PX))
        x1 = int(min(video.width, max(ba[2], bb[2]) + BOX_PAD_PX))
        y1 = int(min(video.height, max(ba[3], bb[3]) + BOX_PAD_PX))
        bd, whole = wmq.box_diff(a_img, b_img, [x0, y0, x1, y1])
        visible = bd >= PIXEL_CHANGE_MIN and bd >= PIXEL_CHANGE_RATIO * max(whole, 1e-6)
        out["pixel_change"] = {"frames": [k_before, k_after], "box": [x0, y0, x1, y1], "box_diff": bd, "frame_diff": whole, "visible": visible}
        if not visible:
            out["reasons"].append("no rendered change of the actor in the observer video")
        # Rendered height of A: rows of the pre-action box whose pixels
        # changed once A left them (lateral and backward walks only, where
        # the two silhouettes do not overlap much).
        if ev["type"] in ("left", "right", "back"):
            diff = np.abs(a_img.astype(np.int16) - b_img.astype(np.int16)).mean(axis=2)
            bx0, by0 = int(max(0, ba[0] - BOX_PAD_PX)), int(max(0, ba[1] - 3 * BOX_PAD_PX))
            bx1, by1 = int(min(video.width, ba[2] + BOX_PAD_PX)), int(min(video.height, ba[3] + 3 * BOX_PAD_PX))
            rows = np.where((diff[by0:by1, bx0:bx1] > 25).sum(axis=1) >= 3)[0]
            if len(rows) >= 10:
                out["rendered_height_px"] = int(rows[-1] - rows[0] + 1)
                out["rendered_top_px"] = int(by0 + rows[0])
    else:
        out["pixel_change"] = None
        out["reasons"].append("observer video frames missing")
    return out


def montage(rec: dict, actor, observer, boxes: dict, out_path: Path) -> bool:
    (_, info_a, _, vid_a), (_, info_o, _, vid_o) = actor, observer
    ev = rec["event"]
    va, vo = wmq.Video(vid_a), wmq.Video(vid_o)
    wa, wo = window_of(info_a), window_of(info_o)
    fs_a, fe_a = (info_a["events"][0].get("frame_start"), info_a["events"][0].get("frame_end"))
    fs_o, fe_o = ev["frame_start"], ev["frame_end"]
    picks = [("before", fs_a - 4, fs_o - 4), ("during", (fs_a + fe_a) // 2, (fs_o + fe_o) // 2),
             ("after", fe_a + 6, fe_o + 6), ("window end", wa[1] - 2, wo[1] - 2)]
    rows = [[], []]
    for label, ka, ko in picks:
        io, ia = vo.frame(ko), va.frame(ka)
        if io is None or ia is None:
            continue
        io = io.copy()
        b = boxes.get(ko)
        if b:
            cv2.rectangle(io, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 0), 1)
        for img, k, who in ((io, ko, f"observer {info_o['bot_name']}"), (ia, ka, f"actor {info_a['bot_name']}")):
            txt = f"{label} | {who} f{k} | {rec['action']} | dist {rec['observer_dist']:.1f} | A height {rec['observer_view'].get('px_height', {}).get('start', 0):.0f} px"
            small = cv2.resize(img, (640, 360))
            cv2.putText(small, txt, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(small, txt, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            rows[0 if who.startswith("observer") else 1].append(small)
    va.close()
    vo.close()
    if not rows[0]:
        return False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid = np.concatenate([np.concatenate(r, axis=1) for r in rows if r], axis=0)
    return bool(cv2.imwrite(str(out_path), grid))


def analyze(actor, observer) -> dict:
    base_a, info_a, fr_a, vid_a = actor
    rec = {"type": TYPE, "actor": info_a["bot_name"], "observer": observer[1]["bot_name"] if observer else None,
           "action": info_a.get("action"), "observer_dist": info_a.get("observer_dist"),
           "scene_check_passed": bool(info_a.get("scene_check_passed")), "scene_search_attempts": info_a.get("scene_search_attempts"),
           "reasons": [], "event": None}
    if not rec["scene_check_passed"]:
        rec["reasons"].append("scene search failed")
        return rec
    if observer is None:
        rec["reasons"].append("observer recording missing")
        return rec
    base_o, info_o, fr_o, vid_o = observer
    for info, frames in ((info_a, fr_a), (info_o, fr_o)):
        if any("frame_start" not in e for e in info.get("events", [])):
            resolve_event_frames(info, frames)
    ev_a = (info_a.get("events") or [None])[0]
    ev_o = (info_o.get("events") or [None])[0]
    win_a, win_o = window_of(info_a), window_of(info_o)
    if not ev_a or ev_a.get("frame_start") is None or not ev_o or ev_o.get("frame_start") is None or not win_a or not win_o:
        rec["reasons"].append("event or window frames unresolved")
        return rec
    rec["event"] = {k: ev_o.get(k) for k in ("type", "key", "frame_start", "frame_end", "motion_ticks", "displacement", "displacement_blocks",
                                             "displacement_forward", "displacement_left", "max_y_rise", "los_clear", "error", "airborne")}
    rec["event"]["actor_frame_start"], rec["event"]["actor_frame_end"] = ev_a["frame_start"], ev_a["frame_end"]
    rec["window_frames"] = win_o[1] - win_o[0] + 1
    rec["actor_window_frames"] = win_a[1] - win_a[0] + 1
    if not WINDOW_FRAMES[0] <= rec["window_frames"] <= WINDOW_FRAMES[1]:
        rec["reasons"].append(f"window {rec['window_frames']} frames, expected {WINDOW_FRAMES[0]}-{WINDOW_FRAMES[1]}")
    if ev_o.get("error"):
        rec["reasons"].append(f"action error: {ev_o['error']}")
    los = ev_o.get("los_clear") or {}
    if not (los.get("before") and los.get("after")):
        rec["reasons"].append("line of sight from the observer blocked")

    rec["actor_motion"] = actor_motion(info_a, fr_a, ev_a, win_a)
    rec["reasons"] += rec["actor_motion"].pop("reasons")
    vo = wmq.Video(vid_o)
    rec["observer_view"] = observer_view(info_o, fr_o, fr_a, win_o, ev_o, vo)
    rec["reasons"] += rec["observer_view"].pop("reasons")
    act_times = [f["renderTime"] for f in fr_a]
    boxes = {k: project_body(fr_o[k], fr_a[nearest_index(act_times, fr_o[k]["renderTime"])], vo.width, vo.height) for k in range(win_o[0], win_o[1] + 1)}
    vo.close()
    for side, item in (("actor", actor), ("observer", observer)):
        win = window_of(item[1])
        ov_ = overlay_scan(wmq.Video(item[3]))
        in_win = [k for k in ov_["toast_frames"] if win[0] <= k <= win[1]]
        rec[f"{side}_overlay"] = {"toast_frames": in_win, "chat_text_frames": [k for k in ov_["chat_text_frames"] if win[0] <= k <= win[1]]}
        if in_win:
            rec["reasons"].append(f"{side}: toast suspected on {len(in_win)} frames")
    rec["_boxes"] = boxes
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--no-montage", action="store_true")
    args = ap.parse_args()

    records = []
    review_dir = args.root / "actor_motion_review"
    for (inst, ep), bots in sorted(wmq.load(args.root).items()):
        actor = next((v for v in bots.values() if v[1].get("role") == "actor"), None)
        observer = next((v for v in bots.values() if v[1].get("role") == "observer"), None)
        if actor is None:
            print(f"episode {ep} instance {inst}: no actor recording, skipped")
            continue
        rec = analyze(actor, observer)
        rec["episode"] = ep
        rec["instance"] = inst
        rec["recordings"] = {name: f"{TYPE}/videos/{v[0]}.mp4" for name, v in bots.items()}
        boxes = rec.pop("_boxes", {})
        rec["clean"] = rec["scene_check_passed"] and not rec["reasons"]
        if not args.no_montage and rec.get("event") and observer is not None:
            path = review_dir / f"ep{ep}_inst{inst}.png"
            if montage(rec, actor, observer, boxes, path):
                rec["montage"] = str(path.relative_to(args.root))
        records.append(rec)
        dist = f"{rec['observer_dist']:.1f}" if rec.get("observer_dist") is not None else "n/a"
        print(f"episode {ep} instance {inst}: {rec['action']} dist {dist} clean={rec['clean']} {rec['reasons']}")

    out = args.out or (args.root / "quality_manifest_actor_motion.json")
    out.write_text(json.dumps(records, indent=1))
    print(f"\nwrote {len(records)} episode records to {out}\n")

    n = len(records)
    clean = [r for r in records if r["clean"]]
    print(f"episodes {n}, scene_check_passed {sum(r['scene_check_passed'] for r in records)}, clean {len(clean)}, failure rate {(n - len(clean)) / max(n, 1):.0%}")
    why = collections.Counter()
    for r in records:
        for reason in r["reasons"]:
            why[" ".join(w for w in reason.split() if not any(c.isdigit() for c in w))] += 1
    for key, cnt in why.most_common():
        print(f"  not clean: {cnt:3d} x {key}")
    print(f"\n{'ep/inst':8s} {'action':8s} {'dist':>4s} {'win':>4s} {'pre':>4s} {'move':>4s} {'post':>4s} {'fwd':>6s} {'left':>6s} {'rise':>5s} "
          f"{'hpx s/min/max/e':>18s} {'rend':>5s} {'out':>4s} {'pix':>5s} {'los':>4s} {'clean':>5s}")
    for r in records:
        if not r.get("event"):
            print(f"{r['episode'][-3:]}/{r['instance'][-1]:4s} {str(r['action']):8s} {str(r['observer_dist']):>4s} {r['reasons']}")
            continue
        am, ov_ = r["actor_motion"], r["observer_view"]
        h = ov_.get("px_height") or {}
        hpx = f"{h.get('start', 0):.0f}/{h.get('min', 0):.0f}/{h.get('max', 0):.0f}/{h.get('end', 0):.0f}" if h else "n/a"
        pix = (ov_.get("pixel_change") or {}).get("box_diff")
        los = r["event"].get("los_clear") or {}
        print(f"{r['episode'][-3:]}/{r['instance'][-1]:4s} {r['action']:8s} {r['observer_dist']:4.1f} {r['window_frames']:4d} "
              f"{am['static_before']['frames']:4d} {am['motion_frames']:4d} {am['static_after']['frames']:4d} "
              f"{am['forward_blocks']:6.2f} {am['left_blocks']:6.2f} {am['max_y_rise']:5.2f} {hpx:>18s} "
              f"{str(ov_.get('rendered_height_px', '-')):>5s} {len(ov_.get('frames_out_of_frame', [])):4d} "
              f"{(f'{pix:.1f}' if pix is not None else 'n/a'):>5s} {str(los.get('before'))[:1] + str(los.get('after'))[:1]:>4s} {str(r['clean'])[:1]:>5s}")
    print("\npixel height of the actor in the observer frame at the window start, by distance (rounded to 0.5 blocks):")
    by_dist = collections.defaultdict(list)
    rendered = collections.defaultdict(list)
    for r in records:
        h = (r.get("observer_view") or {}).get("px_height")
        if h:
            key = round(2 * r["observer_dist"]) / 2
            by_dist[key].append(h["start"])
            if (r.get("observer_view") or {}).get("rendered_height_px"):
                rendered[key].append(r["observer_view"]["rendered_height_px"])
    for d in sorted(by_dist):
        v = by_dist[d]
        rv = rendered.get(d, [])
        focal = (720 / 2.0) / math.tan(math.radians(wmq.VFOV_DEG) / 2.0)
        print(f"  {d:4.1f} blocks: projected {statistics.median(v):6.1f} px (n={len(v)}, {min(v):.0f}-{max(v):.0f}); "
              f"nominal {BODY_HEIGHT * focal / d:6.1f} px; rendered (diff-based, lateral/back walks, includes the name tag) "
              f"{(f'{statistics.median(rv):.0f} px (n={len(rv)})' if rv else 'n/a')}")
    for a in ("forward", "back", "left", "right", "jump", "sneak"):
        sub = [r for r in records if r.get("action") == a]
        if sub:
            print(f"[{a:7s}] episodes {len(sub)}, clean {sum(r['clean'] for r in sub)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

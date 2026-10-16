#!/usr/bin/env python3
"""Per-episode quality manifest for the worldModEval dataset.

Input: a flattened dataset root (see flatten_overhead_dataset.py) holding
worldModEval/{trajectories,episode_info,videos}. The two recordings of an
episode (actor A and observer B) are paired by episode number and instance.

For every event of every episode it checks

  world_changed   the server confirmed the block at the event coordinates
                  changed from the expected pre-state to the expected
                  post-state (rcon `execute if block`), and the actor's
                  client saw the same
  observer_seen   the observer's client received the block update
  static          in both recordings the pose (x/y/z/yaw/pitch) is unchanged
                  and no movement/camera action is pressed during the
                  STATIC_FRAMES frames before frame_start, during the event,
                  and the STATIC_FRAMES frames after frame_end
  in_frame        the event cell centre, projected with the GT pose at
                  frame_start (mineflayer convention: yaw 0 = north = -z,
                  counter-clockwise; pitch positive = up; eye 1.62 above the
                  feet; vertical FOV 70 deg), lies inside the video frame
  los_clear       line of sight from the eye to the cell, computed by the
                  controller through the world at event time
  visible         in_frame and los_clear
  pixel_change    mean absolute pixel difference inside the projected cell
                  box between the frame before the event and a few frames
                  after it, versus the whole-frame difference (a rendered
                  change is visible when the box changes far more than the
                  rest of the frame)

and per episode that the static window spans WINDOW_FRAMES frames.

Output: <root>/quality_manifest_world_mod.json and a summary on stdout.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import cv2
import numpy as np

from traj_segments import resolve_event_frames

TYPE = "worldModEval"
STATIC_FRAMES = 15
WINDOW_FRAMES = (105, 129)
EYE_HEIGHT = 1.62
VFOV_DEG = 70.0
FRAME_MARGIN = 0.02  # fraction of the frame kept clear at the border
POSE_EPS_BLOCKS = 0.01
POSE_EPS_DEG = 0.1
# A forced look leaves a sub-0.1 deg float residue in the camera action of
# the next frame; anything below that is "no camera motion".
CAMERA_EPS_RAD = 2e-3
# The camera video and the trajectory are aligned to within several frames
# (up to ~6 observed), so the rendered change can land before or after the
# client's block update: compare frames VIDEO_SLACK_FRAMES before/after the
# event (still inside the static holds) and locate the first video frame on
# which the cell box changes.
VIDEO_SLACK_FRAMES = 8
PIXEL_CHANGE_MIN = 8.0
# The actor's own arm swing and water/leaf animation raise the whole-frame
# difference, so the box only needs to change twice as much as the frame.
PIXEL_CHANGE_RATIO = 2.0
MOVE_KEYS = ("forward", "back", "left", "right", "jump", "sneak", "sprint")
DEG = 180.0 / math.pi

STEM_RE = re.compile(r"^(?P<prefix>.+?)_(?P<ep>\d{6})_(?P<bot>[A-Za-z]+)_instance_(?P<inst>\d{3})$")


def camera_axes(yaw: float, pitch: float):
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    f = (-sy * cp, sp, -cy * cp)
    r = (cy, 0.0, -sy)
    u = (sy * sp, cp, cy * sp)  # r x f
    return f, r, u


def project(pose: dict, point, width: int, height: int):
    """Pixel coordinates (x right, y down) of a world point, or None when it
    is behind the camera."""
    f, r, u = camera_axes(pose["yaw"], pose["pitch"])
    v = (point[0] - pose["x"], point[1] - (pose["y"] + EYE_HEIGHT), point[2] - pose["z"])
    zc = sum(a * b for a, b in zip(v, f))
    if zc <= 1e-6:
        return None
    xc = sum(a * b for a, b in zip(v, r))
    yc = sum(a * b for a, b in zip(v, u))
    focal = (height / 2.0) / math.tan(math.radians(VFOV_DEG) / 2.0)
    return (width / 2.0 + xc / zc * focal, height / 2.0 - yc / zc * focal)


def in_frame(px, width, height):
    if px is None:
        return False
    mx, my = width * FRAME_MARGIN, height * FRAME_MARGIN
    return mx <= px[0] <= width - mx and my <= px[1] <= height - my


def pose_static(frames, lo, hi):
    """Max pose deviation and pressed actions over frames[lo:hi+1]
    (inclusive). Returns (ok, detail)."""
    lo, hi = max(lo, 0), min(hi, len(frames) - 1)
    if hi < lo:
        return False, {"frames": 0}
    win = frames[lo: hi + 1]
    ref = win[0]
    dpos = max(max(abs(f["x"] - ref["x"]), abs(f["y"] - ref["y"]), abs(f["z"] - ref["z"])) for f in win)
    dyaw = max(abs((((f["yaw"] - ref["yaw"]) + math.pi) % (2 * math.pi)) - math.pi) for f in win) * DEG
    dpitch = max(abs(f["pitch"] - ref["pitch"]) for f in win) * DEG
    keys = sorted({k for f in win for k in MOVE_KEYS if f["action"].get(k)})
    cam = max(max(abs(c) for c in f["action"].get("camera", [0, 0])) for f in win)
    ok = (
        hi - lo + 1 >= 1
        and dpos <= POSE_EPS_BLOCKS
        and dyaw <= POSE_EPS_DEG
        and dpitch <= POSE_EPS_DEG
        and not keys
        and cam <= CAMERA_EPS_RAD
    )
    return ok, {"frames": hi - lo + 1, "dpos": dpos, "dyaw_deg": dyaw, "dpitch_deg": dpitch, "keys": keys, "camera_max": cam}


class Video:
    def __init__(self, path: Path):
        self.cap = cv2.VideoCapture(str(path)) if path and path.exists() else None
        self.n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) if self.cap and self.cap.isOpened() else 0
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if self.n else 1280
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if self.n else 720
        self.cache = {}

    def frame(self, k: int):
        if not self.n:
            return None
        k = max(0, min(k, self.n - 1))
        if k not in self.cache:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, k)
            ok, img = self.cap.read()
            self.cache[k] = img if ok else None
        return self.cache[k]

    def close(self):
        if self.cap:
            self.cap.release()


def box_diff(a, b, box):
    x0, y0, x1, y1 = box
    diff = np.abs(a.astype(np.int16) - b.astype(np.int16)).mean(axis=2)
    return float(diff[y0:y1, x0:x1].mean()), float(diff.mean())


def pixel_change(video: Video, pose: dict, cell, fs: int, fe: int):
    """Mean |diff| inside the projected cell box between the frame
    VIDEO_SLACK_FRAMES before frame_start and the frame VIDEO_SLACK_FRAMES
    after frame_end, vs the whole frame; plus the first video frame in that
    span on which the box differs from the pre-event frame (change_frame) and
    its offset from frame_start (lag_frames, negative = video leads)."""
    k0, k1 = fs - VIDEO_SLACK_FRAMES, fe + VIDEO_SLACK_FRAMES
    a, b = video.frame(k0), video.frame(k1)
    if a is None or b is None:
        return None
    corners = [(cell[0] + dx, cell[1] + dy, cell[2] + dz) for dx in (0, 1) for dy in (0, 1) for dz in (0, 1)]
    pts = [project(pose, c, video.width, video.height) for c in corners]
    if any(p is None for p in pts):
        return None
    x0 = int(max(0, min(p[0] for p in pts)))
    x1 = int(min(video.width, max(p[0] for p in pts) + 1))
    y0 = int(max(0, min(p[1] for p in pts)))
    y1 = int(min(video.height, max(p[1] for p in pts) + 1))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    box = [x0, y0, x1, y1]
    bd, whole = box_diff(a, b, box)
    visible = bd >= PIXEL_CHANGE_MIN and bd >= PIXEL_CHANGE_RATIO * max(whole, 1e-6)
    change_frame = None
    if visible:
        for k in range(k0 + 1, k1 + 1):
            f = video.frame(k)
            if f is None:
                break
            d, _ = box_diff(a, f, box)
            if d >= 0.5 * bd:
                change_frame = k
                break
    return {"box": box, "box_diff": bd, "frame_diff": whole, "visible": visible,
            "change_frame": change_frame,
            "lag_frames": (change_frame - fs) if change_frame is not None else None}


def analyze_recording(info: dict, frames: list, video: Video, events: list) -> dict:
    """Per-bot checks of one recording against the (shared) event list."""
    rec = {"bot": info["bot_name"], "role": info.get("role"), "num_frames": len(frames), "events": []}
    if any("frame_start" not in ev for ev in info.get("events", [])):
        resolve_event_frames(info, frames)
    own = {ev["idx"]: ev for ev in info.get("events", [])}
    bot = info["bot_name"]
    for ev in events:
        mine = own.get(ev["idx"], {})
        fs, fe = mine.get("frame_start"), mine.get("frame_end")
        out = {"idx": ev["idx"], "type": ev["type"], "frame_start": fs, "frame_end": fe, "reasons": []}
        rec["events"].append(out)
        if fs is None or fe is None:
            out["reasons"].append("event not in recording")
            continue
        ok_b, det_b = pose_static(frames, fs - STATIC_FRAMES, fs - 1)
        ok_d, det_d = pose_static(frames, fs, fe)
        ok_a, det_a = pose_static(frames, fe + 1, fe + STATIC_FRAMES)
        out["static_before"], out["static_during"], out["static_after"] = det_b, det_d, det_a
        if det_b["frames"] < STATIC_FRAMES or not ok_b:
            out["reasons"].append("not static before event")
        if not ok_d:
            out["reasons"].append("pose changed during event")
        if det_a["frames"] < STATIC_FRAMES or not ok_a:
            out["reasons"].append("not static after event")
        pose = frames[fs]
        centre = (ev["pos"][0] + 0.5, ev["pos"][1] + 0.5, ev["pos"][2] + 0.5)
        px = project(pose, centre, video.width, video.height)
        out["pixel"] = px
        out["in_frame"] = in_frame(px, video.width, video.height)
        out["los_clear"] = (ev.get("los_clear") or {}).get(bot)
        out["visible"] = bool(out["in_frame"] and out["los_clear"])
        if not out["in_frame"]:
            out["reasons"].append("target outside frame")
        if not out["los_clear"]:
            out["reasons"].append("target occluded")
        out["pixel_change"] = pixel_change(video, pose, ev["pos"], fs, fe)
        out["clean"] = not out["reasons"]
    w0, w1 = info.get("window_start_frame"), info.get("window_end_frame")
    rec["window_frames"] = (w1 - w0 + 1) if w0 is not None and w1 is not None else None
    return rec


def analyze_episode(actor_info, actor_frames, actor_video, obs_info, obs_frames, obs_video) -> dict:
    events = actor_info.get("events", [])
    rec = {
        "type": TYPE,
        "actor": actor_info["bot_name"],
        "observer": obs_info["bot_name"] if obs_info else None,
        "world_type": actor_info.get("world_type"),
        "scene_check_passed": bool(actor_info.get("scene_check_passed")),
        "num_events": len(events),
        "event_types": [e["type"] for e in events],
        "view_angle_deg": actor_info.get("view_angle_deg"),
        "view_angle_band": actor_info.get("view_angle_band"),
        "actor_dist": actor_info.get("actor_dist"),
        "observer_dist": actor_info.get("observer_dist"),
        "reasons": [],
        "events": [],
    }
    if not rec["scene_check_passed"]:
        rec["reasons"].append("scene search failed")
    if not events:
        rec["reasons"].append("no events")
    seen = {}
    for upd in (obs_info or {}).get("observer_block_updates", []):
        seen.setdefault(upd["idx"], []).append(upd)
    for ev in events:
        after = ev["block"] if ev["type"] == "place" else "air"
        e = {
            "idx": ev["idx"], "type": ev["type"], "block": ev["block"], "pos": ev["pos"],
            "duration_ms": (ev.get("end_time_ms") or 0) - (ev.get("start_time_ms") or 0),
            "error": ev.get("error"),
            "server_before_ok": ev.get("server_block_before_ok"),
            "server_after_ok": ev.get("server_block_after_ok"),
            "client_block_after": ev.get("client_block_after"),
            "world_changed": bool(ev.get("world_changed")) and bool(ev.get("server_block_before_ok")),
            "observer_seen": any(u.get("new_block") == after for u in seen.get(ev["idx"], [])),
            "turn_ticks": ev.get("turn_ticks"),
            "reasons": [],
        }
        if not e["world_changed"]:
            e["reasons"].append("world state not changed")
        if obs_info and not e["observer_seen"]:
            e["reasons"].append("observer client did not see the change")
        rec["events"].append(e)
    rec["actor_recording"] = analyze_recording(actor_info, actor_frames, actor_video, events)
    rec["observer_recording"] = (
        analyze_recording(obs_info, obs_frames, obs_video, events) if obs_info else None
    )
    if obs_info is None:
        rec["reasons"].append("observer recording missing")
    for side in ("actor_recording", "observer_recording"):
        r = rec[side]
        if r is None:
            continue
        for e, re_ in zip(rec["events"], r["events"]):
            e[f"{side.split('_')[0]}_visible"] = re_.get("visible")
            e[f"{side.split('_')[0]}_pixel_visible"] = (re_.get("pixel_change") or {}).get("visible")
            e["reasons"] += [f"{side.split('_')[0]}: {x}" for x in re_["reasons"]]
    wf = rec["actor_recording"]["window_frames"]
    rec["window_frames"] = wf
    if wf is None or not WINDOW_FRAMES[0] <= wf <= WINDOW_FRAMES[1]:
        rec["reasons"].append(f"window {wf} frames, expected {WINDOW_FRAMES[0]}-{WINDOW_FRAMES[1]}")
    for e in rec["events"]:
        e["clean"] = not e["reasons"]
    rec["events_clean"] = sum(e["clean"] for e in rec["events"])
    rec["clean"] = bool(events) and not rec["reasons"] and all(e["clean"] for e in rec["events"])
    return rec


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
    ap.add_argument("--root", required=True, type=Path, help="flattened dataset root (one subdir per type)")
    ap.add_argument("--out", type=Path, default=None, help="manifest path (default: <root>/quality_manifest_world_mod.json)")
    args = ap.parse_args()

    records = []
    for (inst, ep), bots in sorted(load(args.root).items()):
        actor = next((v for v in bots.values() if v[1].get("role") == "actor"), None)
        if actor is None:
            print(f"episode {ep} instance {inst}: no actor recording, skipped")
            continue
        observer = next((v for v in bots.values() if v[1].get("role") == "observer"), None)
        av = Video(actor[3])
        ov = Video(observer[3]) if observer else Video(None)
        rec = analyze_episode(actor[1], actor[2], av, observer[1] if observer else None,
                              observer[2] if observer else None, ov)
        av.close()
        ov.close()
        rec["episode"] = ep
        rec["instance"] = inst
        rec["recordings"] = {name: f"{TYPE}/videos/{v[0]}.mp4" for name, v in bots.items()}
        records.append(rec)

    out = args.out or (args.root / "quality_manifest_world_mod.json")
    out.write_text(json.dumps(records, indent=1))
    print(f"wrote {len(records)} episode records to {out}\n")

    n = len(records)
    passed = [r for r in records if r["scene_check_passed"]]
    print(f"episodes {n}, scene_check_passed {len(passed)}, clean {sum(r['clean'] for r in records)}")
    wf = sorted(r["window_frames"] for r in passed if r["window_frames"] is not None)
    if wf:
        print(f"window frames: min {wf[0]} med {wf[len(wf)//2]} max {wf[-1]}, in {WINDOW_FRAMES}: {sum(WINDOW_FRAMES[0] <= w <= WINDOW_FRAMES[1] for w in wf)}/{len(wf)}")
    bands = {}
    for r in records:
        key = "-".join(str(b) for b in r["view_angle_band"]) if r.get("view_angle_band") else "unknown"
        bands.setdefault(key, []).append(r)
    for key, rs in sorted(bands.items()):
        angles = sorted(round(r["view_angle_deg"]) for r in rs if r["scene_check_passed"])
        print(f"view angle band {key} deg: episodes {len(rs)}, scene_check_passed {len(angles)}, clean {sum(r['clean'] for r in rs)}, "
              f"angles {angles}")
    events = [e for r in passed for e in r["events"]]
    if events:
        def rate(key, sub):
            vals = [e.get(key) for e in sub]
            known = [v for v in vals if v is not None]
            return f"{sum(bool(v) for v in known)}/{len(known)}"
        print(f"\n{'events':10s} {'n':>3s} {'changed':>8s} {'obs_seen':>8s} {'A_vis':>7s} {'B_vis':>7s} {'A_pix':>7s} {'B_pix':>7s} {'clean':>6s}")
        for label, sub in [("all", events), ("place", [e for e in events if e["type"] == "place"]), ("break", [e for e in events if e["type"] == "break"])]:
            print(f"{label:10s} {len(sub):3d} {rate('world_changed', sub):>8s} {rate('observer_seen', sub):>8s} "
                  f"{rate('actor_visible', sub):>7s} {rate('observer_visible', sub):>7s} "
                  f"{rate('actor_pixel_visible', sub):>7s} {rate('observer_pixel_visible', sub):>7s} {rate('clean', sub):>6s}")
        why = {}
        for e in events:
            for reason in e["reasons"]:
                key = " ".join(w for w in reason.split() if not any(c.isdigit() for c in w))
                why[key] = why.get(key, 0) + 1
        for r in records:
            for reason in r["reasons"]:
                key = "episode: " + " ".join(w for w in reason.split() if not any(c.isdigit() for c in w))
                why[key] = why.get(key, 0) + 1
        for key, cnt in sorted(why.items(), key=lambda kv: -kv[1]):
            print(f"  not clean: {cnt:3d} x {key}")
        lags = {}
        for r in passed:
            for side in ("actor_recording", "observer_recording"):
                if r[side] is None:
                    continue
                for e in r[side]["events"]:
                    lag = (e.get("pixel_change") or {}).get("lag_frames")
                    if lag is not None:
                        lags.setdefault(side.split("_")[0], []).append(lag)
        for side, vals in lags.items():
            vals.sort()
            print(f"video change frame - frame_start ({side}): min {vals[0]} med {vals[len(vals)//2]} max {vals[-1]} (n={len(vals)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

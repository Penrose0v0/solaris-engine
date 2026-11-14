#!/usr/bin/env python3
"""Annotation, quality manifest and review images for worldModRevisitEval
("EP": one agent interacts with the target cell, yaws out 120 deg, holds
off-screen for H frames, yaws back) and worldModOffscreenEval ("DSP": A
interacts while B looks 90 deg away, after H frames B turns onto the target).

Input: a flattened dataset root holding <type>/{trajectories,episode_info,
videos} for one or both types.

Step 1, annotation (written back into every episode_info; all frame numbers
are indices into that recording's own trajectory / video):

  phases             the script phases with start_frame / end_frame. Every
                     recorded frame carries one script step (frame_trace,
                     matched to the trajectory by renderTime), so the frames
                     are exact.
  event              trigger_frame (first frame at or after the click),
                     action_frames (frames whose action has place_block /
                     mine), effect_frame (first frame rendered at or after
                     the server confirmed the new block state; for B the
                     same instant on B's frames), video_effect_frame (first
                     video frame on which the target box is closer to the end
                     picture than to the start picture; actor recordings).
  target_visibility  GT frustum of every trajectory frame (vertical FOV 70,
                     16:9, eye 1.62 above the feet): the target cell counts
                     as visible when one of its 8 corners or its centre is
                     inside the frame (no occlusion test; line of sight was
                     checked by the controller at setup). visible_intervals,
                     fully_visible_intervals, and for DSP the same for B in
                     both recordings (b_*), the offset of the target from
                     B's view centre after the turn, and the off-screen
                     latency (B's first sight - effect).

Step 2, checks per recording (reasons listed when one fails):
  * scene built, hook did not time out, every script step of the window
    appears exactly once (no recorder skip), phases have their planned length;
  * GT: the server confirmed the pre-state, the change, and the changed
    state at the end; the actor's client agrees; (DSP) B's client received
    the block update and holds the new state at the end;
  * the effect frame lies inside the interact phase; place_block / mine
    appear only inside it, no other key is ever pressed;
  * the actor holds the expected item in the main hand on every frame,
    off-hand empty, hotbar slot constant; (DSP) B's hands are empty;
  * pose: no translation, pitch 0, yaw equals the script on every frame;
  * action stream: one action per frame, action.camera of frame i equals
    pose(i) - pose(i-1) on every frame of the recording;
  * visibility: actor sees the target at the interaction; EP: the target is
    out of view from the end of turn_out to the start of turn_back; DSP: the
    target is outside B's view on every frame from the window start to the
    start of b_turn (and at the effect instant), and within
    MAX_CENTRE_OFFSET_DEG of B's view centre in static_post;
  * video: frame count equals the trajectory; the picture is still inside
    the start static segment and inside the end static segment; where the
    pose is the same at start and end (EP, DSP actor) the two pictures
    differ only inside the target box; the target box does change.

Output: <root>/quality_manifest_world_mod_offscreen.json, review images in
<root>/world_mod_offscreen_review/ and a summary.
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

from co_observe_quality import project_many, visible_many

TYPES = {"worldModRevisitEval": "EP", "worldModOffscreenEval": "DSP"}
NAME_RE = re.compile(r"_(\d{6})_(\w+?)_instance_(\d{3})$")
W, H = 1280, 720
VFOV = 70.0
TRACE_MATCH_MS = 6
MAX_POS_DEV = 0.02
MAX_PITCH_DEG = 0.3
MAX_YAW_ERR_DEG = 0.3
MAX_ACTION_ERR_RAD = 0.0025
MAX_CENTRE_OFFSET_DEG = 5.0
SCALE = 0.5
HUD_TOP_FRACTION = 0.80
BOX_PAD = 10              # half-resolution pixels around the projected target cell
VIDEO_SETTLE = 10         # frames after a turn before the picture is required to be still
STILL_START_MAX = 1.5
STILL_END_MAX = 2.5
STILL_STEP_MAX = 3.0
OUTSIDE_MAX = 4.0         # start vs end picture outside the target box (clouds drift)
BOX_CHANGE_MIN = 8.0      # start vs end picture inside the target box
REVIEW_LAG = 3
MOVE_KEYS = ("place_block", "mine")


def wrap(a):
    return (np.asarray(a) + math.pi) % (2 * math.pi) - math.pi


def intervals(mask) -> list:
    out, start = [], None
    for i, v in enumerate(list(mask) + [False]):
        if v and start is None:
            start = i
        elif not v and start is not None:
            out.append([start, i - 1])
            start = None
    return out


def cell_points(cell) -> np.ndarray:
    x, y, z = cell
    pts = [(x + dx, y + dy, z + dz) for dx in (0, 1) for dy in (0, 1) for dz in (0, 1)]
    pts.append((x + 0.5, y + 0.5, z + 0.5))
    return np.array(pts, float)


def visibility(frames: list, pts: np.ndarray) -> np.ndarray:
    """(n_frames,) fraction of the test points inside the GT frustum."""
    out = np.zeros(len(frames))
    cache: dict = {}
    for i, f in enumerate(frames):
        key = (f["x"], f["y"], f["z"], f["yaw"], f["pitch"])
        if key not in cache:
            cache[key] = float(visible_many(f, pts, W, H).mean())
        out[i] = cache[key]
    return out


def body_points(pos) -> np.ndarray:
    x, y, z = pos
    return np.array([(x + dx, y + dy, z + dz) for dx in (-0.3, 0.3) for dz in (-0.3, 0.3) for dy in (0.0, 0.9, 1.8)], float)


def steps_by_frame(info: dict, frames: list) -> np.ndarray:
    rt = np.array([f["renderTime"] for f in frames], float)
    steps = np.full(len(frames), -1, int)
    for t, s, _deg in info.get("frame_trace") or []:
        i = int(np.searchsorted(rt, t - 0.5))
        if i < len(rt) and rt[i] - t <= TRACE_MATCH_MS:
            steps[i] = s
    return steps


def first_frame_at(frames: list, t_ms) -> int | None:
    if t_ms is None:
        return None
    rt = np.array([f["renderTime"] for f in frames], float)
    i = int(np.searchsorted(rt, t_ms))
    return i if i < len(rt) else None


def load_video(path: Path):
    cap = cv2.VideoCapture(str(path))
    grey, colour = [], []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        small = cv2.resize(f, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)
        colour.append(small)
        grey.append(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32))
    cap.release()
    return grey, colour


def item_name(f, key):
    it = (f.get("inventory") or {}).get(key)
    return it.get("name") if isinstance(it, dict) else None


class Rec:
    def __init__(self, stem: str, base: Path, kind: str):
        self.stem, self.kind = stem, kind
        self.info_path = base / "episode_info" / f"{stem}_episode_info.json"
        self.traj_path = base / "trajectories" / f"{stem}.json"
        self.video_path = base / "videos" / f"{stem}.mp4"
        self.info = json.loads(self.info_path.read_text())
        self.frames = json.loads(self.traj_path.read_text()) if self.traj_path.exists() else None
        self.bot = self.info.get("bot_name")
        self.reasons: list[str] = []
        self.q: dict = {}
        self.colour: list = []

    @property
    def usable(self) -> bool:
        return bool(self.frames) and bool(self.info.get("scene_check_passed")) and bool(self.info.get("frame_trace"))

    @property
    def is_actor(self) -> bool:
        return self.kind == "EP" or self.info.get("agent") == "A"


def annotate_and_check(r: Rec) -> None:
    info, frames, q, reasons = r.info, r.frames, r.q, r.reasons
    plan = info["phase_plan"]
    nwin = plan[-1]["end_step"] + 1
    steps = steps_by_frame(info, frames)
    r.steps = steps
    in_win = steps[(steps >= 0) & (steps < nwin)]
    missing = sorted(set(range(nwin)) - set(in_win.tolist()))
    dup = int(len(in_win) - len(set(in_win.tolist())))
    q.update(window_frames_planned=nwin, window_frames_recorded=int(len(in_win)), recorder_skipped_steps=len(missing), recorder_duplicate_steps=dup)
    if info.get("hook_timed_out"):
        reasons.append("frame hook timed out")
    if missing:
        reasons.append(f"recorder skipped {len(missing)} frame(s) (steps {missing[:6]})")
    if dup:
        reasons.append(f"{dup} doubled step(s)")
    if [f["frame_count"] for f in frames] != list(range(frames[0]["frame_count"], frames[0]["frame_count"] + len(frames))) or any("action" not in f for f in frames):
        reasons.append("action stream and frames are not one-to-one")

    phases = []
    for p in plan:
        idx = np.nonzero((steps >= p["start_step"]) & (steps <= p["end_step"]))[0] if p["frames"] else np.array([], int)
        ph = dict(p, start_frame=int(idx[0]) if len(idx) else None, end_frame=int(idx[-1]) if len(idx) else None, frames_recorded=int(len(idx)))
        phases.append(ph)
        if len(idx) != p["frames"]:
            reasons.append(f"{p['name']} {len(idx)} frames (planned {p['frames']})")
    r.phases = {p["name"]: p for p in phases}
    info["phases"] = phases
    win = np.nonzero((steps >= 0) & (steps < nwin))[0]
    r.win = win
    if not len(win):
        reasons.append("no window frames")
        return
    info["window_start_frame"], info["window_end_frame"] = int(win[0]), int(win[-1])
    q["phase_frames"] = {p["name"]: [p["start_frame"], p["end_frame"]] for p in phases}
    q["lead_frames"], q["tail_frames"] = int(win[0]), int(len(frames) - 1 - win[-1])
    w = slice(int(win[0]), int(win[-1]) + 1)

    # Pose against the script
    pos = np.array([[f["x"], f["y"], f["z"]] for f in frames])
    yaw = np.array([f["yaw"] for f in frames], float)
    pitch = np.array([f["pitch"] for f in frames], float)
    q["pos_dev_max"] = float(np.abs(pos[w] - pos[w][0]).max())
    q["pitch_abs_max_deg"] = float(np.degrees(np.abs(pitch[w]).max()))
    sign = 1.0 if info["direction"] == "left" else -1.0
    deg_by_step = {s: d for _t, s, d in info["frame_trace"]}
    exp = np.array([info["start_yaw"] + sign * math.radians(deg_by_step[s]) if s in deg_by_step else np.nan for s in steps])
    q["yaw_err_max_deg"] = float(np.nanmax(np.degrees(np.abs(wrap(yaw - exp)))[w]))
    q["turn_reached_deg"] = float(np.degrees(np.abs(wrap(yaw - info["start_yaw"])))[w].max())
    q["end_yaw_offset_deg"] = float(np.degrees(wrap(yaw[w][-1] - info["start_yaw"])))
    d = np.degrees(np.abs(wrap(np.diff(yaw))))
    q["yaw_step_histogram_deg"] = dict(sorted(collections.Counter(float(round(v)) for v in d[d > 0.05]).items()))
    if q["pos_dev_max"] > MAX_POS_DEV:
        reasons.append(f"moved {q['pos_dev_max']:.3f} blocks")
    if q["pitch_abs_max_deg"] > MAX_PITCH_DEG:
        reasons.append(f"pitch {q['pitch_abs_max_deg']:.2f} deg")
    if q["yaw_err_max_deg"] > MAX_YAW_ERR_DEG:
        reasons.append(f"yaw off the script by {q['yaw_err_max_deg']:.2f} deg")

    # Action stream against the GT pose, every frame of the recording
    cam = np.array([f["action"]["camera"] for f in frames], float)
    e_yaw = np.abs(wrap(cam[1:, 0] - wrap(np.diff(yaw))))
    e_pitch = np.abs(cam[1:, 1] - np.diff(pitch))
    q["action_vs_pose_max_err_rad"] = float(max(e_yaw.max(), e_pitch.max()))
    q["action_camera_sum_deg"] = float(np.degrees(cam[w][1:, 0].sum()))
    q["pose_yaw_change_deg"] = float(np.degrees(np.unwrap(yaw[w])[-1] - yaw[w][0]))
    if q["action_vs_pose_max_err_rad"] > MAX_ACTION_ERR_RAD:
        reasons.append(f"action.camera differs from the pose step by {q['action_vs_pose_max_err_rad']:.4f} rad")
    dt = np.diff([f["renderTime"] for f in frames])
    q["frame_dt_ms_min_max"] = [float(dt.min()), float(dt.max())]

    # Interaction: keys, effect frame, hands
    ev = info.get("event") or {}
    it = r.phases["interact"]
    key = "place_block" if info["interaction"] == "place" else "mine"
    keys = [k for k in frames[0]["action"] if k != "camera"]
    pressed = {k: [i for i, f in enumerate(frames) if f["action"].get(k)] for k in keys}
    pressed = {k: v for k, v in pressed.items() if v}
    q["keys_pressed_frames"] = pressed
    ev["trigger_frame"] = first_frame_at(frames, ev.get("trigger_time_ms"))
    ev["effect_frame"] = first_frame_at(frames, ev.get("effect_time_ms"))
    ev["effect_frame_note"] = "first frame of THIS recording rendered at or after the server confirmed the new block state"
    if r.is_actor:
        ev["action_frames"] = pressed.get(key, [])
        ev["action_key"] = key
        if set(pressed) - {key}:
            reasons.append(f"unexpected keys pressed: {sorted(set(pressed) - {key})}")
        if not pressed.get(key):
            reasons.append(f"no {key} in the action stream")
        elif it["start_frame"] is not None and not (it["start_frame"] <= pressed[key][0] and pressed[key][-1] <= it["end_frame"]):
            reasons.append(f"{key} outside the interact phase: frames {pressed[key][0]}..{pressed[key][-1]}")
    elif pressed:
        reasons.append(f"observer pressed keys: {sorted(pressed)}")
    for name, ok in (("pre-state not confirmed by the server", ev.get("server_block_before_ok")),
                     ("change not confirmed (server / actor client)", ev.get("world_changed")),
                     ("changed state not there at the end", ev.get("server_block_end_ok"))):
        if not ok:
            reasons.append(name)
    if ev.get("error"):
        reasons.append(f"interaction error: {ev['error']}")
    after = info["block"] if info["interaction"] == "place" else "air"
    if r.is_actor and ev.get("client_block_end") != after:
        reasons.append(f"actor client holds {ev.get('client_block_end')} at the end")
    if not r.is_actor:
        ups = info.get("own_client_block_updates") or []
        q["observer_block_updates"] = len(ups)
        q["observer_client_block_end"] = info.get("observer_client_block_end")
        if not any(u.get("new_block") == after for u in ups):
            reasons.append("observer client never received the block update")
        if info.get("observer_client_block_end") != after:
            reasons.append(f"observer client holds {info.get('observer_client_block_end')} at the end")
    ef = ev["effect_frame"]
    q["effect_frame"], q["trigger_frame"] = ef, ev["trigger_frame"]
    if ef is None or it["start_frame"] is None or not (it["start_frame"] <= ef <= it["end_frame"]):
        reasons.append(f"effect frame {ef} outside the interact phase {[it['start_frame'], it['end_frame']]}")
    else:
        q["effect_offset_in_interact"] = ef - it["start_frame"]
        nxt = r.phases["turn_out" if r.kind == "EP" else "b_turn"]["start_frame"]
        q["static_frames_after_effect"] = None if nxt is None else nxt - ef - 1
    info["event"] = ev
    info["events"] = [ev]
    held = collections.Counter(item_name(f, "heldItem") for f in frames)
    off = collections.Counter(item_name(f, "offHandItem") for f in frames)
    slots = collections.Counter((f.get("inventory") or {}).get("quickBarSlot") for f in frames)
    expected = info.get("held_item") if r.is_actor else None
    q["held_items"] = {str(k): v for k, v in held.items()}
    if set(held) != {expected} or set(off) != {None} or len(slots) != 1:
        reasons.append(f"hands not as expected: held {dict(held)}, off-hand {dict(off)}, slots {dict(slots)}")

    # Target visibility from the GT frustum
    pts = cell_points(info["target_cell"])
    r.frac = visibility(frames, pts)
    px, py, _ = project_many(frames[int(win[0])], pts, W, H)
    r.box0 = [float(px[:8].min()), float(py[:8].min()), float(px[:8].max()), float(py[:8].max())]


def target_offset(frame: dict, cell) -> tuple[float, float]:
    """(horizontal, vertical) angle in degrees of the cell centre from the view centre."""
    dx, dy, dz = cell[0] + 0.5 - frame["x"], cell[1] + 0.5 - (frame["y"] + 1.62), cell[2] + 0.5 - frame["z"]
    yaw = math.atan2(-dx, -dz)
    pitch = math.atan2(dy, math.hypot(dx, dz))
    return float(np.degrees(wrap(yaw - frame["yaw"]))), float(np.degrees(pitch - frame["pitch"]))


def visibility_annotate(recs: list) -> None:
    method = ("GT frustum of every trajectory frame (vertical FOV 70 deg, 16:9, eye 1.62 above the feet); the target cell is visible "
              "when one of its 8 corners or its centre is inside the frame; no occlusion test")
    for r in recs:
        tv = {"method": method, "frames_are": f"indices into this recording ({r.bot})",
              "visible_intervals": intervals(r.frac > 0), "fully_visible_intervals": intervals(r.frac == 1),
              "own_visible_fraction_by_frame": [round(float(v), 3) for v in r.frac]}
        it = r.phases["interact"]
        if r.is_actor and it["start_frame"] is not None and r.frac[it["start_frame"]: it["end_frame"] + 1].min() < 1:
            r.reasons.append("target not fully in the actor's view during the interaction")
        if r.kind == "EP":
            a, b = r.phases["turn_out"]["end_frame"], r.phases["turn_back"]["start_frame"]
            if a is not None and b is not None:
                tv["hidden_at_full_turn"] = bool(r.frac[a: b].max() == 0) if b > a else bool(r.frac[a] == 0)
                hid = intervals(r.frac[r.win[0]: r.win[-1] + 1] == 0)
                tv["hidden_interval"] = [hid[0][0] + int(r.win[0]), hid[0][1] + int(r.win[0])] if hid else None
                tv["hidden_frames"] = None if not hid else hid[0][1] - hid[0][0] + 1
                if not tv["hidden_at_full_turn"]:
                    r.reasons.append("target still in view at the full turn")
            off = target_offset(r.frames[int(r.win[-1])], r.info["target_cell"])
            tv["target_offset_from_view_centre_end_deg"] = off
        r.info["target_visibility"] = tv
    if recs[0].kind != "DSP" or len(recs) != 2:
        return
    a = next(r for r in recs if r.is_actor)
    b = next(r for r in recs if not r.is_actor)
    rt_b = np.array([f["renderTime"] for f in b.frames], float)
    bt = b.phases["b_turn"]
    vis_b = intervals(b.frac > 0)
    first_b = vis_b[0][0] if vis_b else None
    eff_b = b.info["event"].get("effect_frame")
    hidden_until_turn = bool(bt["start_frame"] is not None and b.frac[: bt["start_frame"]].max() == 0)
    hidden_at_effect = bool(eff_b is not None and b.frac[max(0, eff_b - 1): eff_b + 2].max() == 0)
    post = b.phases["static_post"]
    off = target_offset(b.frames[post["start_frame"]], b.info["target_cell"]) if post["start_frame"] is not None else (None, None)
    px, py, _ = project_many(b.frames[post["start_frame"] or -1], cell_points(b.info["target_cell"])[8:], W, H)
    body = visibility(b.frames, body_points(a.info["start_pos"]))
    b_body = visibility(a.frames, body_points(b.info["start_pos"]))
    for r in recs:
        tv = r.info["target_visibility"]
        to_own = (lambda i: i) if r is b else (lambda i: None if i is None else int(np.argmin(np.abs(np.array([f["renderTime"] for f in r.frames]) - rt_b[i]))))
        tv.update({
            "b": b.bot,
            "b_frames_note": "b_* frames are indices into B's recording; *_own are the same instants in this recording (by renderTime)",
            "b_visible_intervals": vis_b,
            "b_fully_visible_intervals": intervals(b.frac == 1),
            "b_hidden_interval": [0, first_b - 1] if first_b else None,
            "b_first_visible_frame": first_b,
            "b_first_visible_frame_own": to_own(first_b),
            "b_first_fully_visible_frame": (intervals(b.frac == 1) or [[None]])[0][0],
            "b_effect_frame": eff_b,
            "offscreen_frames_after_effect": None if first_b is None or eff_b is None else first_b - eff_b,
            "hidden_from_b_until_turn": hidden_until_turn,
            "hidden_from_b_at_effect": hidden_at_effect,
            "target_offset_from_b_view_centre_end_deg": list(off),
            "target_centre_pixel_in_b_end": [float(px[0]), float(py[0])],
            "actor_body_in_b_view_intervals": intervals(body > 0),
            "b_body_in_actor_view_intervals": intervals(b_body > 0),
        })
        r.q["target_visibility"] = {k: v for k, v in tv.items() if k not in ("method", "frames_are", "own_visible_fraction_by_frame", "b_frames_note")}
    if not hidden_until_turn:
        b.reasons.append("target inside B's view before B's turn")
    if not hidden_at_effect:
        b.reasons.append("target inside B's view at the effect")
    if off[0] is None or math.hypot(*off) > MAX_CENTRE_OFFSET_DEG:
        b.reasons.append(f"target {off} deg off B's view centre after the turn")
    if post["start_frame"] is None or b.frac[post["start_frame"]: post["end_frame"] + 1].min() < 1:
        b.reasons.append("target not fully in B's view in static_post")


def video_checks(r: Rec) -> None:
    q, reasons = r.q, r.reasons
    grey, r.colour = load_video(r.video_path)
    q["video_frames"] = len(grey)
    if len(grey) != len(r.frames):
        reasons.append(f"video has {len(grey)} frames, trajectory {len(r.frames)}")
        return
    rows = int(grey[0].shape[0] * HUD_TOP_FRACTION)
    move = np.array([0.0] + [float(np.abs(grey[i][:rows] - grey[i - 1][:rows]).mean()) for i in range(1, len(grey))])

    def still(a, b):
        if a is None or b is None or b - a < 2:
            return None, None
        return float(np.abs(grey[a][:rows] - grey[b][:rows]).mean()), float(move[a + 1: b + 1].max())

    pre, post = r.phases["static_pre"], r.phases["static_post"]
    turned = r.kind == "EP" or not r.is_actor
    q["video_still_start"], q["video_still_start_step_max"] = still(pre["start_frame"], pre["end_frame"])
    end0 = None if post["start_frame"] is None else post["start_frame"] + (VIDEO_SETTLE if turned else 2)
    q["video_still_end_frames"] = [end0, len(grey) - 1]
    q["video_still_end"], q["video_still_end_step_max"] = still(end0, len(grey) - 1)
    for name, lim in (("start", STILL_START_MAX), ("end", STILL_END_MAX)):
        span, step = q[f"video_still_{name}"], q[f"video_still_{name}_step_max"]
        if span is None or span > lim or step > STILL_STEP_MAX:
            reasons.append(f"picture not still in the {name} static segment (span {span}, step {step})")
    if pre["end_frame"] is None or end0 is None:
        return
    if r.kind == "DSP" and not r.is_actor:
        return
    # Same pose at start and end: the pictures may differ only at the target.
    x0, y0, x1, y1 = [v * SCALE for v in r.box0]
    bx = (max(0, int(x0) - BOX_PAD), max(0, int(y0) - BOX_PAD), min(grey[0].shape[1], int(x1) + BOX_PAD), min(rows, int(y1) + BOX_PAD))
    ref = grey[max(pre["start_frame"], pre["end_frame"] - 2)]
    endp = grey[min(len(grey) - 1, end0 + 2)]
    d = np.abs(ref - endp)[:rows]
    mask = np.ones(d.shape, bool)
    mask[bx[1]: bx[3], bx[0]: bx[2]] = False
    q["target_box_px"] = [round(v, 1) for v in r.box0]
    q["start_vs_end_outside_target"] = float(d[mask].mean())
    q["start_vs_end_outside_changed_fraction"] = float((d[mask] > 12).mean())
    q["start_vs_end_inside_target"] = float(d[~mask].mean())
    if q["start_vs_end_outside_target"] > OUTSIDE_MAX:
        reasons.append(f"start and end pictures differ outside the target ({q['start_vs_end_outside_target']:.2f})")
    if q["start_vs_end_inside_target"] < BOX_CHANGE_MIN:
        reasons.append(f"target box does not change in the video ({q['start_vs_end_inside_target']:.2f})")
    inner = (slice(max(0, int(y0)), min(rows, int(y1))), slice(max(0, int(x0)), int(x1)))
    box_d = [float(np.abs(g[inner] - ref[inner]).mean()) for g in grey]
    # First visible change (a dig shows cracks and the swinging tool from its
    # first frame) and the first frame on which the box is closer to the end
    # picture than to the start picture (the block appeared / disappeared).
    first = next((i for i in range(pre["end_frame"] - 2, len(grey)) if box_d[i] > BOX_CHANGE_MIN / 2), None)
    v = next((i for i in range(pre["end_frame"] - 2, len(grey)) if float(np.abs(grey[i][inner] - endp[inner]).mean()) < box_d[i]), None)
    r.info["event"]["video_first_change_frame"] = first
    q["video_first_change_frame"] = first
    r.info["event"]["video_effect_frame"] = v
    q["video_effect_frame"] = v
    q["video_effect_lag_frames"] = None if v is None or q.get("effect_frame") is None else v - q["effect_frame"]
    r.diff_img = np.clip(np.abs(ref - endp) * 4, 0, 255).astype(np.uint8)


def tile(r: Rec, i: int, label: str) -> np.ndarray:
    if not r.colour or i is None:
        return np.zeros((int(H * SCALE), int(W * SCALE), 3), np.uint8)
    i = max(0, min(len(r.colour) - 1, i))
    t = r.colour[i].copy()
    lab = f"{r.bot} | {label} f{i} | target in view {r.frac[i]:.2f}"
    cv2.putText(t, lab, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4)
    cv2.putText(t, lab, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return t


def review_image(recs: list, out: Path, title: str) -> None:
    rows = []
    for r in sorted(recs, key=lambda r: (not r.is_actor, r.bot)):
        ph = r.phases
        mid = lambda p: None if ph[p]["start_frame"] is None else (ph[p]["start_frame"] + ph[p]["end_frame"]) // 2 + REVIEW_LAG  # noqa: E731
        ef = r.q.get("effect_frame")
        tiles = [tile(r, mid("static_pre"), "static_pre"), tile(r, None if ef is None else ef + 12, "after effect")]
        if r.kind == "EP":
            tiles.append(tile(r, ph["turn_out"]["end_frame"] + REVIEW_LAG if ph["turn_out"]["end_frame"] is not None else None, "turned out"))
        else:
            tiles.append(tile(r, ph["b_turn"]["start_frame"], "before B's turn"))
        tiles.append(tile(r, (ph["static_post"]["end_frame"] or 0) - 2, "static_post"))
        if getattr(r, "diff_img", None) is not None:
            tiles.append(cv2.cvtColor(r.diff_img, cv2.COLOR_GRAY2BGR))
        elif r.kind == "DSP":
            tiles.append(np.zeros_like(tiles[0]))
        rows.append(np.hstack(tiles))
    img = np.vstack(rows)
    bar = np.zeros((30, img.shape[1], 3), np.uint8)
    cv2.putText(bar, title, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), np.vstack([bar, img]), [cv2.IMWRITE_JPEG_QUALITY, 90])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--no-review", action="store_true")
    ap.add_argument("--no-write-info", action="store_true")
    args = ap.parse_args()

    records = []
    for type_name, kind in TYPES.items():
        base = args.root / type_name
        if not (base / "episode_info").is_dir():
            continue
        groups = collections.defaultdict(list)
        for info_path in sorted((base / "episode_info").glob("*_episode_info.json")):
            stem = info_path.name[: -len("_episode_info.json")]
            m = NAME_RE.search(stem)
            if not m:
                continue
            ep, bot, inst = m.groups()
            # EP: every recording is its own episode; DSP: the two recordings pair up
            groups[(inst, ep, bot if kind == "EP" else "")].append(Rec(stem, base, kind))
        for (inst, ep, _), recs in sorted(groups.items()):
            need = 1 if kind == "EP" else 2
            usable = len(recs) == need and all(r.usable for r in recs)
            if usable:
                for r in recs:
                    annotate_and_check(r)
                usable = all(len(r.win) for r in recs)
            if usable:
                visibility_annotate(recs)
                for r in recs:
                    if r.video_path.exists():
                        video_checks(r)
                    else:
                        r.reasons.append("video missing")
                    if not args.no_write_info:
                        r.info_path.write_text(json.dumps(r.info, indent=2))
                if not args.no_review:
                    i0 = recs[0].info
                    name = f"{kind}_ep{ep}_inst{inst}{'_' + recs[0].bot if kind == 'EP' else ''}_{i0['interaction']}_H{i0['hold_frames']}_{i0['direction']}.jpg"
                    review_image(recs, args.root / "world_mod_offscreen_review" / name,
                                 f"{kind} episode {ep} instance {inst}: {i0['interaction']} {i0.get('block')}, H {i0['hold_frames']}, turn {i0['direction']}")
            for r in recs:
                if not usable and not r.reasons:
                    if len(recs) != need:
                        r.reasons.append("other bot's recording missing")
                    elif not r.info.get("scene_check_passed"):
                        r.reasons.append("scene setup failed")
                    elif not r.frames:
                        r.reasons.append("trajectory missing")
                    elif not r.info.get("frame_trace"):
                        r.reasons.append("no frame trace")
                    else:
                        r.reasons.append("other bot's recording unusable")
                i = r.info
                records.append({
                    "type": type_name, "eval_kind": kind, "recording": r.stem, "episode": ep, "instance": inst, "bot": r.bot,
                    "video": f"{type_name}/videos/{r.stem}.mp4", "agent": i.get("agent", "A"), "interaction": i.get("interaction"),
                    "hold_frames": i.get("hold_frames"), "direction": i.get("direction"), "block": i.get("block"),
                    "target_cell": i.get("target_cell"), "cell": f"{kind}/{i.get('interaction')}/H{i.get('hold_frames')}/{i.get('direction')}",
                    "n_frames": len(r.frames or []), **r.q, "reasons": r.reasons, "clean": not r.reasons,
                    "_group": f"{type_name}/{inst}/{ep}/{r.bot if kind == 'EP' else ''}",
                })
    by_ep = collections.defaultdict(list)
    for rec in records:
        by_ep[rec.pop("_group")].append(rec)
    for pair in by_ep.values():
        ok = all(rec["clean"] for rec in pair) and len(pair) == (1 if pair[0]["eval_kind"] == "EP" else 2)
        for rec in pair:
            rec["episode_clean"] = ok

    out = args.out or (args.root / "quality_manifest_world_mod_offscreen.json")
    out.write_text(json.dumps(records, indent=1))
    f = lambda v, n=2: "-" if v is None else f"{v:.{n}f}"  # noqa: E731
    print(f"{'ep/inst/bot':13s} {'cell':24s} {'ag':2s} {'phase lengths':22s} {'skip':>4s} {'eff':>4s} {'+int':>4s} {'vlag':>4s} {'yawerr':>6s} {'act err':>7s} {'still s/e':>10s} {'out/in':>11s} {'B first':>7s} {'offscr':>6s} {'B ctr':>11s} clean")
    for rec in records:
        ph = rec.get("phase_frames") or {}
        lens = "/".join(str((v[1] - v[0] + 1) if v[0] is not None else 0) for v in ph.values())
        tv = rec.get("target_visibility") or {}
        ctr = tv.get("target_offset_from_b_view_centre_end_deg")
        print(f"{rec['episode'][-3:]}/{rec['instance'][-1]}/{rec['bot']:6s} {rec['cell']:24s} {rec['agent']:2s} {lens:22s} {str(rec.get('recorder_skipped_steps', '-')):>4s} "
              f"{str(rec.get('effect_frame')):>4s} {str(rec.get('effect_offset_in_interact')):>4s} {str(rec.get('video_effect_lag_frames')):>4s} "
              f"{f(rec.get('yaw_err_max_deg')):>6s} {f(rec.get('action_vs_pose_max_err_rad'), 4):>7s} {f(rec.get('video_still_start')) + '/' + f(rec.get('video_still_end')):>10s} "
              f"{f(rec.get('start_vs_end_outside_target')) + '/' + f(rec.get('start_vs_end_inside_target'), 1):>11s} {str(tv.get('b_first_visible_frame')):>7s} "
              f"{str(tv.get('offscreen_frames_after_effect')):>6s} {('-' if not ctr or ctr[0] is None else f'{ctr[0]:+.1f},{ctr[1]:+.1f}'):>11s} {rec['clean']} {rec['reasons'] or ''}")
    eps = [v[0] for v in by_ep.values()]
    clean = [v for v in eps if v["episode_clean"]]
    print(f"\nwrote {len(records)} records to {out}; episodes {len(eps)}, clean episodes {len(clean)}")
    print("clean episodes by cell:", dict(sorted(collections.Counter(v["cell"] for v in clean).items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

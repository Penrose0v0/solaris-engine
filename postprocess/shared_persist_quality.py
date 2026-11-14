#!/usr/bin/env python3
"""Annotation, quality manifest and review images for sharedPersistEval
(shared-world persistence with a conditioning frame: the first bot sees the
north wall x in the conditioning frame and turns away; after a gap in which
nobody sees x the second bot turns to x).

Input: a flattened dataset root holding sharedPersistEval/{trajectories,
episode_info,videos}.

Step 1, annotation (written back into every episode_info; all frame numbers
are indices into that recording's own trajectory / video):

  phases           the five script phases with start_frame / end_frame (exact:
                   one script step per recorded frame, matched by renderTime).
  condition_frame  = window_start_frame, the first frame of static_pre.
  x_visibility     from the GT frustum (pose of every trajectory frame,
                   vertical FOV 70, 16:9; a block of x is visible when one of
                   its corners or its inner face centre is inside the frame;
                   no occlusion test, the courtyard is convex):
                     x_last_visible_frame     last frame the first bot sees any of x
                     x_first_revisible_frame  first frame the second bot sees any of x
                     x_unobserved_frames, gap_phase_x_hidden_from_both, ...
                   Events of the other bot are mapped to this recording's
                   frames by renderTime; `*_native` gives bot and frame in the
                   recording whose GT pose produced them.

Step 2, checks per recording:
  * scene built, both recordings present, hook did not time out;
  * every script step of the window appears exactly once;
  * phase lengths as planned (10 / 33 / G / 33 / 15);
  * the bot does not translate, pitch 0, yaw follows the script;
  * action stream: action.camera of frame i equals pose(i) - pose(i-1) for
    every frame; one action per frame;
  * conditioning frame: x in the first bot's view (>= 90 % of its blocks),
    outside the second bot's view; the second bot has the first one 40 deg
    off its view centre (inside the frame), the first bot does not have the
    second one in its frame; bots 6-7 blocks apart on an east-west line;
  * x outside both views for every frame of the gap phase and of the whole
    span between the last and the first re-visible frame; the second bot
    does not see x before its turn;
  * north wall overlap: IoU of the blocks of x in the first bot's
    conditioning frame and in the second bot's final observation;
  * video: frame count equals the trajectory (= action) count; the picture is
    still in the start static segment (static_pre) and in the end static
    segment (second_observe + tail); picture motion starts / stops with the
    bot's own turn (lag in frames).

Per episode a review image (rows A / B; columns condition frame, gap,
second_observe) goes to <root>/shared_persist_review/.

Output: <root>/quality_manifest_shared_persist.json and a summary.
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

from cross_revisit_quality import (DIST_RANGE, H, HUD_TOP_FRACTION, MAX_ACTION_ERR_RAD, MAX_PITCH_DEG, MAX_POS_DEV, MAX_YAW_ERR_DEG,
                                   MIN_IOU, MIN_OBSERVE_FRACTION, MOVE_MIN, PARTNER_OFFSET_DEG, PARTNER_TOL_DEG, REVIEW_LAG, SCALE,
                                   STILL_END_MAX, STILL_START_MAX, STILL_STEP_MAX, VIDEO_LAG_RANGE, VIDEO_SETTLE, W,
                                   block_visibility, load_video, nearest_frame, steps_by_frame, wrap, x_points)

TYPE = "sharedPersistEval"
KEY = "shared_persist"
NAME_RE = re.compile(r"_(\d{6})_(\w+?)_instance_(\d{3})$")
HALF_HFOV_DEG = math.degrees(math.atan(math.tan(math.radians(35.0)) * W / H))  # 51.2
PARTNER_OUT_MARGIN_DEG = 10.0  # the first bot's partner must be this far outside its frame edge


class Rec:
    def __init__(self, stem: str, base: Path):
        self.stem = stem
        self.info_path = base / "episode_info" / f"{stem}_episode_info.json"
        self.traj_path = base / "trajectories" / f"{stem}.json"
        self.video_path = base / "videos" / f"{stem}.mp4"
        self.info = json.loads(self.info_path.read_text())
        self.frames = json.loads(self.traj_path.read_text()) if self.traj_path.exists() else None
        self.bot = self.info.get("bot_name")
        self.reasons: list[str] = []
        self.q: dict = {}

    @property
    def usable(self) -> bool:
        return bool(self.frames) and bool(self.info.get("scene_check_passed")) and bool(self.info.get("frame_trace")) and KEY in self.info


def annotate_and_check(r: Rec) -> None:
    info, frames, q, reasons = r.info, r.frames, r.q, r.reasons
    sp = info[KEY]
    me = info["bots"][r.bot]
    r.agent = me["agent"]
    r.is_first = r.agent == sp["first_observer_agent"]
    steps = steps_by_frame(info, frames)
    r.steps = steps
    nwin = sp["window_frames"]

    in_win = steps[(steps >= 0) & (steps < nwin)]
    missing = sorted(set(range(nwin)) - set(in_win.tolist()))
    dup = int(len(in_win) - len(set(in_win.tolist())))
    q["window_frames_planned"] = nwin
    q["window_frames_recorded"] = int(len(in_win))
    q["recorder_skipped_steps"] = len(missing)
    q["recorder_duplicate_steps"] = dup
    if info.get("hook_timed_out"):
        reasons.append("frame hook timed out")
    if missing:
        reasons.append(f"recorder skipped {len(missing)} frame(s) (steps {missing[:6]})")
    q["n_actions"] = sum(1 for f in frames if "action" in f)
    if len(frames) != len({f["frame_count"] for f in frames}) or q["n_actions"] != len(frames):
        reasons.append("action stream and frames differ in length")
    if dup:
        reasons.append(f"{dup} doubled step(s)")

    phases = []
    for p in info["phase_plan"]:
        idx = np.nonzero((steps >= p["start_step"]) & (steps <= p["end_step"]))[0]
        ph = dict(p)
        ph["start_frame"] = int(idx[0]) if len(idx) else None
        ph["end_frame"] = int(idx[-1]) if len(idx) else None
        ph["frames_recorded"] = int(len(idx))
        for k in ("turning", "observing"):
            a = p.get(f"{k}_agent")
            ph[f"{k}_bot"] = None if a is None else sp["agents"][a]
        phases.append(ph)
        if len(idx) != p["frames"]:
            reasons.append(f"{p['name']} {len(idx)} frames (planned {p['frames']})")
    r.phases = {p["name"]: p for p in phases}
    info["phases"] = phases
    win = np.nonzero((steps >= 0) & (steps < nwin))[0]
    if len(win):
        info["window_start_frame"], info["window_end_frame"] = int(win[0]), int(win[-1])
        info["condition_frame"] = int(win[0])
    q["phase_frames"] = {p["name"]: [p["start_frame"], p["end_frame"]] for p in phases}
    q["lead_frames"] = int(win[0]) if len(win) else None
    q["tail_frames"] = int(len(frames) - 1 - win[-1]) if len(win) else None

    pos = np.array([[f["x"], f["y"], f["z"]] for f in frames])
    yaw = np.array([f["yaw"] for f in frames], float)
    pitch = np.array([f["pitch"] for f in frames], float)
    w = slice(int(win[0]), int(win[-1]) + 1) if len(win) else slice(0, 0)
    q["pos_dev_max"] = float(np.abs(pos[w] - pos[w][0]).max()) if len(win) else None
    q["pitch_abs_max_deg"] = float(np.degrees(np.abs(pitch[w]).max())) if len(win) else None
    sign = 1.0 if sp["turn_direction"] == "left" else -1.0
    deg_by_step = {s: d for _t, s, d in info["frame_trace"]}
    exp = np.array([me["yaw0"] + sign * math.radians(deg_by_step[s]) if s in deg_by_step else np.nan for s in steps])
    err = np.degrees(np.abs(wrap(yaw - exp)))
    q["yaw_err_max_deg"] = float(np.nanmax(err[w])) if len(win) else None
    q["turn_reached_deg"] = float(np.degrees(np.abs(wrap(yaw - me["yaw0"])))[w].max()) if len(win) else None
    q["start_azimuth_deg"] = float((-math.degrees(yaw[w][0])) % 360) if len(win) else None
    q["end_azimuth_deg"] = float((-math.degrees(yaw[w][-1])) % 360) if len(win) else None
    d = np.degrees(wrap(np.diff(yaw)))
    moving = d[np.abs(d) > 0.05]
    q["yaw_step_histogram_deg"] = {str(k): v for k, v in sorted(collections.Counter(float(round(v)) for v in moving).items())}
    if q["pos_dev_max"] is None or q["pos_dev_max"] > MAX_POS_DEV:
        reasons.append(f"moved {q['pos_dev_max']} blocks")
    if q["pitch_abs_max_deg"] is None or q["pitch_abs_max_deg"] > MAX_PITCH_DEG:
        reasons.append(f"pitch {q['pitch_abs_max_deg']} deg")
    if q["yaw_err_max_deg"] is None or q["yaw_err_max_deg"] > MAX_YAW_ERR_DEG:
        reasons.append(f"yaw off the script by {q['yaw_err_max_deg']} deg")
    dw = d[int(win[0]): int(win[-1])] if len(win) else d[:0]
    if (dw[np.abs(dw) > 0.05] * sign < 0).any():
        reasons.append("a yaw step against the scripted turn direction")

    cam = np.array([f["action"]["camera"] for f in frames], float)
    e_yaw = np.abs(wrap(cam[1:, 0] - wrap(np.diff(yaw))))
    e_pitch = np.abs(cam[1:, 1] - np.diff(pitch))
    q["action_vs_pose_max_err_rad"] = float(max(e_yaw.max(), e_pitch.max()))
    q["action_camera_sum_deg"] = float(np.degrees(cam[w][1:, 0].sum())) if len(win) else None
    q["pose_yaw_change_deg"] = float(np.degrees(wrap(yaw[w][-1] - yaw[w][0]))) if len(win) else None
    keys = [k for k in frames[0]["action"] if k != "camera"]
    q["other_actions_pressed"] = sorted({k for f in frames[w] for k in keys if f["action"][k]})
    if q["action_vs_pose_max_err_rad"] > MAX_ACTION_ERR_RAD:
        reasons.append(f"action.camera differs from the pose step by {q['action_vs_pose_max_err_rad']:.4f} rad")
    if q["other_actions_pressed"]:
        reasons.append(f"keys pressed: {q['other_actions_pressed']}")
    dt = np.diff([f["renderTime"] for f in frames])
    q["frame_dt_ms_min_max"] = [float(dt.min()), float(dt.max())]

    pts, n = x_points(info)
    r.vis = block_visibility(frames, pts, n)
    r.frac = r.vis.mean(axis=1)


def video_checks(r: Rec, colour_out: dict) -> None:
    q, reasons = r.q, r.reasons
    grey, colour = load_video(r.video_path)
    colour_out[r.bot] = colour
    q["video_frames"] = len(grey)
    if len(grey) != len(r.frames):
        reasons.append(f"video has {len(grey)} frames, trajectory {len(r.frames)}")
        return
    rows = int(grey[0].shape[0] * HUD_TOP_FRACTION)
    move = np.array([0.0] + [float(np.abs(grey[i][:rows] - grey[i - 1][:rows]).mean()) for i in range(1, len(grey))])
    lags = {}
    for name, p in r.phases.items():
        if p["turning_bot"] != r.bot or p["start_frame"] is None:
            continue
        a, b = p["start_frame"], p["end_frame"]
        on = next((i for i in range(max(1, a - 5), min(len(move), b + 15)) if move[i] > MOVE_MIN), None)
        off = next((i for i in range(min(len(move) - 1, b + 15), a, -1) if move[i] > MOVE_MIN), None)
        lags[name] = [None if on is None else on - a, None if off is None else off - b]
    q["video_turn_lag_frames"] = lags
    flat = [v for pair in lags.values() for v in pair]
    if any(v is None or not (VIDEO_LAG_RANGE[0] <= v <= VIDEO_LAG_RANGE[1]) for v in flat):
        reasons.append(f"video turn timing off: {lags}")

    def still(a, b):
        if a is None or b is None or b - a < 2:
            return None, None
        return float(np.abs(grey[a][:rows] - grey[b][:rows]).mean()), float(move[a + 1: b + 1].max())

    pre, obs = r.phases["static_pre"], r.phases["second_observe"]
    q["video_still_start"], q["video_still_start_step_max"] = still(pre["start_frame"], pre["end_frame"])
    end0 = None if obs["start_frame"] is None else obs["start_frame"] + (VIDEO_SETTLE if r.phases["second_turn"]["turning_bot"] == r.bot else 2)
    q["video_still_end_frames"] = [end0, len(grey) - 1]
    q["video_still_end"], q["video_still_end_step_max"] = still(end0, len(grey) - 1)
    for name, lim in (("start", STILL_START_MAX), ("end", STILL_END_MAX)):
        span, step = q[f"video_still_{name}"], q[f"video_still_{name}_step_max"]
        if span is None or span > lim or step > STILL_STEP_MAX:
            reasons.append(f"picture not still in the {name} static segment (span {span}, step {step})")


def pair_annotate(first: Rec, second: Rec) -> None:
    sp = first.info[KEY]
    nwin = sp["window_frames"]
    t_of = lambda r, i: r.frames[i]["renderTime"]  # noqa: E731

    def in_window(r):
        return (r.steps >= 0) & (r.steps < nwin)

    fv = np.nonzero(in_window(first) & (first.frac > 0))[0]
    sv = np.nonzero(in_window(second) & (second.frac > 0))[0]
    ev = {
        "x_last_visible": (first, int(fv[-1]) if len(fv) else None),
        "x_first_revisible": (second, int(sv[0]) if len(sv) else None),
    }
    c1 = first.phases["static_pre"]["start_frame"]
    c2 = second.phases["static_pre"]["start_frame"]
    obs2 = second.phases["second_observe"]
    m2 = None if obs2["start_frame"] is None else (obs2["start_frame"] + obs2["end_frame"]) // 2
    iou = None
    if c1 is not None and m2 is not None:
        a, b = first.vis[c1], second.vis[m2]
        iou = float((a & b).sum() / max(1, (a | b).sum()))

    for r in (first, second):
        other = second if r is first else first
        xv = {
            "method": "GT frustum of every trajectory frame (vertical FOV 70 deg, 16:9, eye 1.62 above the feet); a block of x is "
                      "visible when a corner or its inner face centre is inside the frame; no occlusion test",
            "frames_are": f"indices into this recording ({r.bot}); events of the other bot are mapped by renderTime",
            "first_observer": first.bot,
            "second_observer": second.bot,
            "x_blocks": int(r.vis.shape[1]),
        }
        for name, (src, fr) in ev.items():
            own = None if fr is None else (fr if src is r else nearest_frame(r.frames, t_of(src, fr)))
            xv[f"{name}_frame"] = own
            xv[f"{name}_native"] = None if fr is None else {"bot": src.bot, "frame": fr}
        lo, hi = xv["x_last_visible_frame"], xv["x_first_revisible_frame"]
        xv["x_unobserved_frames"] = None if lo is None or hi is None else hi - lo - 1
        # Conditioning frame (each bot's own first window frame)
        xv["condition_frame_visible_fraction"] = {
            first.bot: None if c1 is None else float(first.frac[c1]),
            second.bot: None if c2 is None else float(second.frac[c2]),
        }
        xv["x_visible_in_first_condition_frame"] = None if c1 is None else bool(first.frac[c1] >= MIN_OBSERVE_FRACTION)
        xv["x_hidden_in_second_condition_frame"] = None if c2 is None else bool(second.frac[c2] == 0)
        # Scripted gap phase, each bot over its own frames of the phase
        g_me, g_other = r.phases["gap"], other.phases["gap"]
        fr_me = r.frac[g_me["start_frame"]: g_me["end_frame"] + 1] if g_me["start_frame"] is not None else np.array([np.nan])
        fr_other = other.frac[g_other["start_frame"]: g_other["end_frame"] + 1] if g_other["start_frame"] is not None else np.array([np.nan])
        xv["gap_phase_max_visible_fraction"] = {r.bot: float(fr_me.max()), other.bot: float(fr_other.max())}
        xv["gap_phase_x_hidden_from_both"] = bool(fr_me.max() == 0 and fr_other.max() == 0)
        # Whole unobserved span (first bot's last sight .. second bot's first sight)
        if ev["x_last_visible"][1] is not None and ev["x_first_revisible"][1] is not None:
            t_lo, t_hi = t_of(first, ev["x_last_visible"][1]), t_of(second, ev["x_first_revisible"][1])
            hidden = True
            for rr in (first, second):
                rt = np.array([f["renderTime"] for f in rr.frames])
                span = (rt > t_lo + 10) & (rt < t_hi - 10)
                hidden = hidden and bool((rr.frac[span] == 0).all())
            xv["unobserved_span_x_hidden_from_both"] = hidden
        # The second bot must not see x before its own turn
        t2 = second.phases["second_turn"]["start_frame"]
        xv["second_sees_x_before_its_turn"] = None if t2 is None else bool(second.frac[:t2].max() > 0)
        xv["visible_fraction_at_final_observe"] = None if m2 is None else float(second.frac[m2])
        xv["north_wall_iou_condition_vs_final"] = iou
        xv["own_visible_fraction_by_frame"] = [round(float(v), 3) for v in r.frac]
        r.info["x_visibility"] = xv
        r.q["x_visibility"] = {k: v for k, v in xv.items() if k not in ("method", "frames_are", "own_visible_fraction_by_frame")}

        if not xv["x_visible_in_first_condition_frame"]:
            r.reasons.append(f"x not (fully) in the first bot's conditioning frame: {xv['condition_frame_visible_fraction']}")
        if not xv["x_hidden_in_second_condition_frame"]:
            r.reasons.append("x visible in the second bot's conditioning frame")
        if xv["x_last_visible_frame"] is None or xv["x_first_revisible_frame"] is None:
            r.reasons.append("an observer never sees x")
        if not xv["gap_phase_x_hidden_from_both"]:
            r.reasons.append(f"x visible during the gap phase: {xv['gap_phase_max_visible_fraction']}")
        if xv.get("unobserved_span_x_hidden_from_both") is not True:
            r.reasons.append("x visible to someone inside the unobserved span")
        if xv["second_sees_x_before_its_turn"] is not False:
            r.reasons.append("the second bot sees x before its turn")
        if xv["visible_fraction_at_final_observe"] is None or xv["visible_fraction_at_final_observe"] < MIN_OBSERVE_FRACTION:
            r.reasons.append(f"x only partly in view in the final observation: {xv['visible_fraction_at_final_observe']}")
        if iou is None or iou < MIN_IOU:
            r.reasons.append(f"north wall overlap too small (IoU {iou})")

    # Start geometry from the GT poses of the conditioning frames
    fa, fb = first.frames[c1 or 0], second.frames[c2 or 0]
    dist = math.hypot(fa["x"] - fb["x"], fa["z"] - fb["z"])
    for r, mine, theirs in ((first, fa, fb), (second, fb, fa)):
        bearing = math.atan2(-(theirs["x"] - mine["x"]), -(theirs["z"] - mine["z"]))  # mineflayer yaw towards the partner
        off = abs(math.degrees(float(wrap(bearing - mine["yaw"]))))
        r.q["partner_distance"] = dist
        r.q["partner_offset_deg"] = off
        r.q["partner_in_condition_frame"] = bool(off < HALF_HFOV_DEG)
        if not (DIST_RANGE[0] <= dist <= DIST_RANGE[1]):
            r.reasons.append(f"partner {dist:.2f} blocks away")
        if abs(mine["z"] - theirs["z"]) > 0.05:
            r.reasons.append("bots not on an east-west line")
        if r is second and abs(off - PARTNER_OFFSET_DEG) > PARTNER_TOL_DEG:
            r.reasons.append(f"second bot: partner {off:.1f} deg off the view centre (expected {PARTNER_OFFSET_DEG})")
        if r is first and off < HALF_HFOV_DEG + PARTNER_OUT_MARGIN_DEG:
            r.reasons.append(f"first bot: partner only {off:.1f} deg off the view centre (should be outside the frame)")
    a_rec = first if first.agent == "A" else second
    b_rec = second if a_rec is first else first
    if not a_rec.frames[0]["x"] < b_rec.frames[0]["x"]:
        a_rec.reasons.append("A is not west of B")


def review_image(recs: dict, colour: dict, out: Path, title: str) -> None:
    order = sorted(recs.values(), key=lambda r: r.agent)
    cols = ["static_pre", "gap", "second_observe"]
    rows = []
    for r in order:
        tiles = []
        for name in cols:
            p = r.phases[name]
            vid = colour.get(r.bot) or []
            if p["start_frame"] is None or not vid:
                tiles.append(np.zeros((int(H * SCALE), int(W * SCALE), 3), np.uint8))
                continue
            i = p["start_frame"] if name == "static_pre" else min(len(vid) - 1, (p["start_frame"] + p["end_frame"]) // 2 + REVIEW_LAG)
            t = vid[i].copy()
            lab = f"{r.agent} {r.bot} | {'condition frame' if name == 'static_pre' else name} f{i} | x in view {r.frac[min(i, len(r.frac) - 1)]:.2f}"
            cv2.putText(t, lab, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4)
            cv2.putText(t, lab, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            tiles.append(t)
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
    ap.add_argument("--no-write-info", action="store_true", help="do not write phases / x_visibility back into episode_info")
    args = ap.parse_args()

    base = args.root / TYPE
    episodes = collections.defaultdict(dict)
    for info_path in sorted((base / "episode_info").glob("*_episode_info.json")):
        stem = info_path.name[: -len("_episode_info.json")]
        m = NAME_RE.search(stem)
        if not m:
            continue
        ep, bot, inst = m.groups()
        episodes[(inst, ep)][bot] = Rec(stem, base)

    records = []
    for (inst, ep), recs in sorted(episodes.items()):
        any_info = next(iter(recs.values())).info
        sp = any_info.get(KEY) or any_info.get(f"{KEY}_planned") or {}
        cell = f"G{sp.get('gap_frames')}/{sp.get('first_observer_agent')}-first"
        usable = len(recs) == 2 and all(r.usable for r in recs.values())
        colour: dict = {}
        if usable:
            for r in recs.values():
                annotate_and_check(r)
            first = next(r for r in recs.values() if r.is_first)
            second = next(r for r in recs.values() if not r.is_first)
            pair_annotate(first, second)
            for r in recs.values():
                if r.video_path.exists():
                    video_checks(r, colour)
                else:
                    r.reasons.append("video missing")
                if not args.no_write_info:
                    r.info_path.write_text(json.dumps(r.info, indent=2))
            if not args.no_review:
                out = args.root / "shared_persist_review" / f"ep{ep}_inst{inst}_G{sp.get('gap_frames')}_{sp.get('first_observer_agent')}first.jpg"
                review_image(recs, colour, out, f"episode {ep} instance {inst}: {cell} ({sp.get('turn_direction')} turns), first {first.bot}, second {second.bot}")
        for bot, r in sorted(recs.items()):
            if not usable:
                if len(recs) != 2:
                    r.reasons.append("other bot's recording missing")
                if not r.info.get("scene_check_passed"):
                    r.reasons.append("scene setup failed")
                elif not r.frames:
                    r.reasons.append("trajectory missing")
                elif not r.info.get("frame_trace"):
                    r.reasons.append("no frame trace")
                if not r.reasons:
                    r.reasons.append("other bot's recording unusable")
            records.append({
                "type": TYPE, "recording": r.stem, "episode": ep, "instance": inst, "bot": bot, "video": f"{TYPE}/videos/{r.stem}.mp4",
                "gap_frames": sp.get("gap_frames"), "first_observer_agent": sp.get("first_observer_agent"),
                "turn_direction": sp.get("turn_direction"), "cell": cell,
                "agent": getattr(r, "agent", None), "observer_role": r.info.get("observer_role"), "n_frames": len(r.frames or []),
                **r.q, "reasons": r.reasons,
            })
    by_ep = collections.defaultdict(list)
    for rec in records:
        by_ep[(rec["instance"], rec["episode"])].append(rec)
    for pair in by_ep.values():
        ok = all(not rec["reasons"] for rec in pair) and len(pair) == 2
        for rec in pair:
            rec["clean"] = not rec["reasons"]
            rec["episode_clean"] = ok

    out = args.out or (args.root / "quality_manifest_shared_persist.json")
    out.write_text(json.dumps(records, indent=1))
    print(f"{'ep/inst/bot':15s} {'cell':12s} {'ag':2s} {'role':6s} {'phase frames':16s} {'skip':>4s} {'yawerr':>6s} {'act err':>7s} {'xlast':>5s} {'xre':>4s} {'unobs':>5s} {'gap hid':>7s} {'IoU':>5s} {'partner':>7s} {'lag':>6s} clean")
    for rec in records:
        ph = rec.get("phase_frames") or {}
        lens = "/".join(str((v[1] - v[0] + 1) if v[0] is not None else "-") for v in ph.values())
        xv = rec.get("x_visibility") or {}
        lag = ";".join(f"{a},{b}" for a, b in (rec.get("video_turn_lag_frames") or {}).values())
        f = lambda v, n=2: "-" if v is None else f"{v:.{n}f}"  # noqa: E731
        print(f"{rec['episode'][-3:]}/{rec['instance'][-1]}/{rec['bot']:8s} {rec['cell']:12s} {str(rec.get('agent')):2s} {str(rec.get('observer_role')):6s} {lens:16s} "
              f"{str(rec.get('recorder_skipped_steps', '-')):>4s} {f(rec.get('yaw_err_max_deg')):>6s} {f(rec.get('action_vs_pose_max_err_rad'), 4):>7s} "
              f"{str(xv.get('x_last_visible_frame')):>5s} {str(xv.get('x_first_revisible_frame')):>4s} {str(xv.get('x_unobserved_frames')):>5s} "
              f"{str(xv.get('gap_phase_x_hidden_from_both')):>7s} {f(xv.get('north_wall_iou_condition_vs_final')):>5s} {f(rec.get('partner_offset_deg'), 1):>7s} {lag:>6s} {rec['clean']} {rec['reasons'] or ''}")
    eps = {k: v[0] for k, v in by_ep.items()}
    clean = [v for v in eps.values() if v["episode_clean"]]
    print(f"\nwrote {len(records)} records to {out}; episodes {len(eps)}, clean episodes {len(clean)}")
    print("clean episodes by cell:", dict(sorted(collections.Counter(v["cell"] for v in clean).items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Annotation, quality manifest and review images for crossRevisitEval
(long-term shared-world consistency: the first observer turns to the north
wall x, looks, turns back; after a gap the second observer turns to x;
control episodes: the first observer keeps looking).

Input: a flattened dataset root holding crossRevisitEval/{trajectories,
episode_info,videos}.

Step 1, annotation (written back into every episode_info, all frame numbers
are indices into that recording's own trajectory / video):

  phases        the seven script phases with start_frame / end_frame. Every
                recorded frame carries one script step (frame_trace, matched
                to the trajectory by renderTime), so the frames are exact,
                not a wall-clock estimate.
  x_visibility  from the GT frustum (pose of every trajectory frame, vertical
                FOV 70, 16:9; a block of x counts as visible when one of its
                corners or its inner face centre is inside the frame; no
                occlusion test, the courtyard is convex):
                  x_first_visible_frame     first observer starts to see x
                  x_last_visible_frame      last frame the first observer sees
                                            any of x (null in control episodes)
                  x_first_revisible_frame   first frame the second observer
                                            sees any of x
                  gap_phase_x_hidden_from_both, x_unobserved_frames, ...
                Events of the other bot are mapped to this recording's frames
                by renderTime (both controllers share the host clock); the
                `*_native` entries give bot and frame in the recording whose
                GT pose produced them.

Step 2, checks per recording:
  * scene built, both recordings present, hook did not time out;
  * every script step of the window appears exactly once (a recorder skip
    or a doubled frame is reported; pose and action stay consistent anyway);
  * phase lengths as planned (10 / 33 / 15 / 33 / G / 33 / 15);
  * the bot does not translate, pitch 0, yaw follows the script;
  * action stream: action.camera of frame i equals pose(i) - pose(i-1) for
    every frame of the recording;
  * start geometry: partner 6-7 blocks away and 40 deg off the view centre,
    x outside both views during static_pre;
  * main group: x outside both views for every frame of the gap phase;
    control: the first observer sees x from first_observe to the end;
  * both observers see (nearly) the same part of x (IoU of visible blocks);
  * video: frame count equals the trajectory (= action) count; the picture
    is still in the start static segment (static_pre) and in the end static
    segment (second_observe + tail); picture motion starts / stops with the
    bot's own turns (lag in frames).

Per episode a review image (rows A / B; columns static_pre, first_observe,
gap, second_observe) goes to <root>/cross_revisit_review/.

Output: <root>/quality_manifest_cross_revisit.json and a summary.
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

from co_observe_quality import visible_many

TYPE = "crossRevisitEval"
NAME_RE = re.compile(r"_(\d{6})_(\w+?)_instance_(\d{3})$")
W, H = 1280, 720
TRACE_MATCH_MS = 6
MAX_POS_DEV = 0.02
MAX_PITCH_DEG = 0.3
MAX_YAW_ERR_DEG = 0.3
MAX_ACTION_ERR_RAD = 0.0025
PARTNER_OFFSET_DEG = 40.0
PARTNER_TOL_DEG = 1.5
DIST_RANGE = (5.95, 7.05)
MIN_OBSERVE_FRACTION = 0.9
MIN_IOU = 0.8
SCALE = 0.5
HUD_TOP_FRACTION = 0.80
MOVE_MIN = 4.0            # mean abs grey difference between consecutive frames while the camera turns
VIDEO_LAG_RANGE = (-2, 10)
REVIEW_LAG = 3
VIDEO_SETTLE = 10         # frames after a turn before its bot's picture is required to be still
STILL_START_MAX = 1.5     # mean abs grey difference between the first and last frame of the start static segment
STILL_END_MAX = 2.5       # ... of the end static segment (the encoder keeps refining after the turn)
STILL_STEP_MAX = 3.0      # largest frame-to-frame difference inside a static segment


def wrap(a):
    return (np.asarray(a) + math.pi) % (2 * math.pi) - math.pi


def x_points(info: dict) -> tuple[np.ndarray, int]:
    """(N*9, 3) test points of the north wall blocks and the block count."""
    court = info["courtyard"]
    names = {m["module"] for m in court["modules"] if m["side"] == "north"}
    blocks = [b for b in court["blocks"] if b["module"] in names]
    pts = []
    for b in blocks:
        x, y, z = b["x"], b["y"], b["z"]
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    pts.append((x + dx, y + dy, z + dz))
        pts.append((x + 0.5, y + 0.5, z + 1.0))  # inner (south) face centre
    return np.array(pts, float), len(blocks)


def block_visibility(frames: list, pts: np.ndarray, n_blocks: int) -> np.ndarray:
    """(n_frames, n_blocks) bool: block inside the GT frustum."""
    out = np.zeros((len(frames), n_blocks), bool)
    cache: dict = {}
    for i, f in enumerate(frames):
        key = (f["x"], f["y"], f["z"], f["yaw"], f["pitch"])
        if key not in cache:
            cache[key] = visible_many(f, pts, W, H).reshape(n_blocks, 9).any(axis=1)
        out[i] = cache[key]
    return out


def steps_by_frame(info: dict, frames: list) -> np.ndarray:
    """Script step of every trajectory frame (-1 outside the trace)."""
    rt = np.array([f["renderTime"] for f in frames], float)
    steps = np.full(len(frames), -1, int)
    for t, s, _deg in info.get("frame_trace") or []:
        i = int(np.searchsorted(rt, t - 0.5))
        if i < len(rt) and rt[i] - t <= TRACE_MATCH_MS:
            steps[i] = s
    return steps


def nearest_frame(frames: list, t_ms: float) -> int:
    rt = np.array([f["renderTime"] for f in frames], float)
    return int(np.argmin(np.abs(rt - t_ms)))


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
        return bool(self.frames) and bool(self.info.get("scene_check_passed")) and bool(self.info.get("frame_trace")) \
            and "cross_revisit" in self.info


def annotate_and_check(r: Rec) -> None:
    """Per-recording part: phases, own visibility, pose / action checks."""
    info, frames, q, reasons = r.info, r.frames, r.q, r.reasons
    cr = info["cross_revisit"]
    me = info["bots"][r.bot]
    r.agent = me["agent"]
    r.is_first = r.agent == cr["first_observer_agent"]
    steps = steps_by_frame(info, frames)
    r.steps = steps
    nwin = cr["window_frames"]

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
    if len(frames) != len({f["frame_count"] for f in frames}) or any("action" not in f for f in frames):
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
            ph[f"{k}_bot"] = None if a is None else ("both" if a == "both" else cr["agents"][a])
        phases.append(ph)
        if len(idx) != p["frames"]:
            reasons.append(f"{p['name']} {len(idx)} frames (planned {p['frames']})")
    r.phases = {p["name"]: p for p in phases}
    info["phases"] = phases
    win = np.nonzero((steps >= 0) & (steps < nwin))[0]
    if len(win):
        info["window_start_frame"], info["window_end_frame"] = int(win[0]), int(win[-1])
    q["phase_frames"] = {p["name"]: [p["start_frame"], p["end_frame"]] for p in phases}
    q["lead_frames"] = int(win[0]) if len(win) else None
    q["tail_frames"] = int(len(frames) - 1 - win[-1]) if len(win) else None

    # Pose
    pos = np.array([[f["x"], f["y"], f["z"]] for f in frames])
    yaw = np.array([f["yaw"] for f in frames], float)
    pitch = np.array([f["pitch"] for f in frames], float)
    w = slice(int(win[0]), int(win[-1]) + 1) if len(win) else slice(0, 0)
    q["pos_dev_max"] = float(np.abs(pos[w] - pos[w][0]).max()) if len(win) else None
    q["pitch_abs_max_deg"] = float(np.degrees(np.abs(pitch[w]).max())) if len(win) else None
    sign = 1.0 if r.agent == "A" else -1.0
    deg_by_step = {s: d for _t, s, d in info["frame_trace"]}
    exp = np.array([me["yaw0"] + sign * math.radians(deg_by_step[s]) if s in deg_by_step else np.nan for s in steps])
    err = np.degrees(np.abs(wrap(yaw - exp)))
    q["yaw_err_max_deg"] = float(np.nanmax(err[w])) if len(win) else None
    turned = np.degrees(np.abs(wrap(yaw - me["yaw0"])))
    q["turn_reached_deg"] = float(turned[w].max()) if len(win) else None
    d = np.degrees(np.abs(wrap(np.diff(yaw))))
    moving = d[d > 0.05]
    q["yaw_step_histogram_deg"] = dict(sorted(collections.Counter(float(round(v)) for v in moving).items()))
    if q["pos_dev_max"] is None or q["pos_dev_max"] > MAX_POS_DEV:
        reasons.append(f"moved {q['pos_dev_max']} blocks")
    if q["pitch_abs_max_deg"] is None or q["pitch_abs_max_deg"] > MAX_PITCH_DEG:
        reasons.append(f"pitch {q['pitch_abs_max_deg']} deg")
    if q["yaw_err_max_deg"] is None or q["yaw_err_max_deg"] > MAX_YAW_ERR_DEG:
        reasons.append(f"yaw off the script by {q['yaw_err_max_deg']} deg")

    # Action stream against the GT pose, every frame of the recording
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
    # Still pictures: the start (static_pre) and the end (second_observe plus
    # the tail, after VIDEO_SETTLE frames for the video lag and the encoder).
    def still(a: int, b: int):
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
    cr = first.info["cross_revisit"]
    control = cr["is_control"]
    nwin = cr["window_frames"]
    t_of = lambda r, i: r.frames[i]["renderTime"]  # noqa: E731

    def in_window(r):
        return (r.steps >= 0) & (r.steps < nwin)

    fv = np.nonzero(in_window(first) & (first.frac > 0))[0]
    sv = np.nonzero(in_window(second) & (second.frac > 0))[0]
    ev = {
        "x_first_visible": (first, int(fv[0]) if len(fv) else None),
        "x_last_visible": (first, None if control or not len(fv) else int(fv[-1])),
        "x_first_revisible": (second, int(sv[0]) if len(sv) else None),
    }
    obs1, obs2 = first.phases["first_observe"], second.phases["second_observe"]
    m1 = None if obs1["start_frame"] is None else (obs1["start_frame"] + obs1["end_frame"]) // 2
    m2 = None if obs2["start_frame"] is None else (obs2["start_frame"] + obs2["end_frame"]) // 2
    iou = None
    if m1 is not None and m2 is not None:
        a, b = first.vis[m1], second.vis[m2]
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
        # Scripted gap phase: both bots' own GT, each over its own frames of the phase
        g_me, g_other = r.phases["gap"], other.phases["gap"]
        fr_me = r.frac[g_me["start_frame"]: g_me["end_frame"] + 1] if g_me["start_frame"] is not None else np.array([np.nan])
        fr_other = other.frac[g_other["start_frame"]: g_other["end_frame"] + 1] if g_other["start_frame"] is not None else np.array([np.nan])
        xv["gap_phase_max_visible_fraction"] = {r.bot: float(fr_me.max()), other.bot: float(fr_other.max())}
        xv["gap_phase_x_hidden_from_both"] = None if control else bool(fr_me.max() == 0 and fr_other.max() == 0)
        # Whole unobserved span (first observer's last sight .. second observer's first sight)
        if not control and ev["x_last_visible"][1] is not None and ev["x_first_revisible"][1] is not None:
            t_lo, t_hi = t_of(first, ev["x_last_visible"][1]), t_of(second, ev["x_first_revisible"][1])
            hidden = True
            for rr in (first, second):
                rt = np.array([f["renderTime"] for f in rr.frames])
                span = (rt > t_lo + 10) & (rt < t_hi - 10)
                hidden = hidden and bool((rr.frac[span] == 0).all())
            xv["unobserved_span_x_hidden_from_both"] = hidden
        if control:
            a = first.phases["first_observe"]["start_frame"]
            b = first.phases["second_observe"]["end_frame"]
            xv["first_observer_sees_x_until_end"] = None if a is None or b is None else bool((first.frac[a: b + 1] > 0).all())
            xv["first_observer_min_visible_fraction_after_turn"] = None if a is None or b is None else float(first.frac[a: b + 1].min())
        xv["visible_fraction_at_observe"] = {
            first.bot: None if m1 is None else float(first.frac[m1]),
            second.bot: None if m2 is None else float(second.frac[m2]),
        }
        xv["observed_blocks_iou"] = iou
        xv["own_visible_fraction_by_frame"] = [round(float(v), 3) for v in r.frac]
        r.info["x_visibility"] = xv
        r.q["x_visibility"] = {k: v for k, v in xv.items() if k not in ("method", "frames_are", "own_visible_fraction_by_frame")}

        # Checks
        pre = r.phases["static_pre"]
        if pre["start_frame"] is not None and r.frac[: pre["end_frame"] + 1].max() > 0:
            r.reasons.append("x visible before the first turn")
        if xv["x_first_revisible_frame"] is None or xv["x_first_visible_frame"] is None:
            r.reasons.append("an observer never sees x")
        if not control:
            if not xv["gap_phase_x_hidden_from_both"]:
                r.reasons.append(f"x visible during the gap phase: {xv['gap_phase_max_visible_fraction']}")
            if xv.get("unobserved_span_x_hidden_from_both") is False:
                r.reasons.append("x visible to someone inside the unobserved span")
        elif not xv.get("first_observer_sees_x_until_end"):
            r.reasons.append("control: the first observer loses x")
        vf = xv["visible_fraction_at_observe"]
        if any(v is None or v < MIN_OBSERVE_FRACTION for v in vf.values()):
            r.reasons.append(f"x only partly in view while observing: {vf}")
        if iou is None or iou < MIN_IOU:
            r.reasons.append(f"observed parts of x overlap too little (IoU {iou})")

    # Start geometry from the GT poses
    fa, fb = first.frames[first.phases["static_pre"]["start_frame"] or 0], second.frames[second.phases["static_pre"]["start_frame"] or 0]
    dist = math.hypot(fa["x"] - fb["x"], fa["z"] - fb["z"])
    for r, mine, theirs in ((first, fa, fb), (second, fb, fa)):
        bearing = math.atan2(-(theirs["x"] - mine["x"]), -(theirs["z"] - mine["z"]))  # mineflayer yaw towards the partner
        off = abs(math.degrees(float(wrap(bearing - mine["yaw"]))))
        r.q["partner_distance"] = dist
        r.q["partner_offset_deg"] = off
        r.q["start_azimuth_deg"] = float((-math.degrees(mine["yaw"])) % 360)
        if not (DIST_RANGE[0] <= dist <= DIST_RANGE[1]):
            r.reasons.append(f"partner {dist:.2f} blocks away")
        if abs(off - PARTNER_OFFSET_DEG) > PARTNER_TOL_DEG:
            r.reasons.append(f"partner {off:.1f} deg off the view centre")
        if abs(mine["z"] - theirs["z"]) > 0.05:
            r.reasons.append("bots not on an east-west line")
    a_rec = first if first.agent == "A" else second
    b_rec = second if a_rec is first else first
    if not a_rec.frames[0]["x"] < b_rec.frames[0]["x"]:
        a_rec.reasons.append("A is not west of B")


def review_image(recs: dict, colour: dict, out: Path, title: str) -> None:
    order = sorted(recs.values(), key=lambda r: r.agent)
    cols = ["static_pre", "first_observe", "gap", "second_observe"]
    rows = []
    for r in order:
        tiles = []
        for name in cols:
            p = r.phases[name]
            vid = colour.get(r.bot) or []
            if p["start_frame"] is None or not vid:
                tiles.append(np.zeros((int(H * SCALE), int(W * SCALE), 3), np.uint8))
                continue
            i = min(len(vid) - 1, (p["start_frame"] + p["end_frame"]) // 2 + REVIEW_LAG)
            t = vid[i].copy()
            lab = f"{r.agent} {r.bot} | {name} f{i} | x in view {r.frac[min(i, len(r.frac) - 1)]:.2f}"
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
        cr = any_info.get("cross_revisit") or any_info.get("cross_revisit_planned") or {}
        group = "control" if cr.get("is_control") else "main"
        cell = f"{group}/G{cr.get('gap_frames')}/{cr.get('first_observer_agent')}-first"
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
                out = args.root / "cross_revisit_review" / f"ep{ep}_inst{inst}_{group}_G{cr.get('gap_frames')}_{cr.get('first_observer_agent')}first.jpg"
                review_image(recs, colour, out, f"episode {ep} instance {inst}: {cell}, first observer {first.bot}, second {second.bot}")
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
                "group": group, "gap_frames": cr.get("gap_frames"), "is_control": cr.get("is_control"),
                "first_observer_agent": cr.get("first_observer_agent"), "cell": cell,
                "agent": getattr(r, "agent", None), "observer_role": r.info.get("observer_role"), "n_frames": len(r.frames or []),
                **r.q, "reasons": r.reasons,
            })
    # an episode is clean only when both recordings are
    by_ep = collections.defaultdict(list)
    for rec in records:
        by_ep[(rec["instance"], rec["episode"])].append(rec)
    for pair in by_ep.values():
        ok = all(not rec["reasons"] for rec in pair) and len(pair) == 2
        for rec in pair:
            rec["clean"] = not rec["reasons"]
            rec["episode_clean"] = ok

    out = args.out or (args.root / "quality_manifest_cross_revisit.json")
    out.write_text(json.dumps(records, indent=1))
    print(f"{'ep/inst/bot':15s} {'cell':20s} {'ag':2s} {'role':6s} {'phase frames':24s} {'skip':>4s} {'yawerr':>6s} {'act err':>7s} {'xlast':>5s} {'xre':>4s} {'unobs':>5s} {'gap hid':>7s} {'IoU':>5s} {'lag':>14s} clean")
    for rec in records:
        ph = rec.get("phase_frames") or {}
        lens = "/".join(str((v[1] - v[0] + 1) if v[0] is not None else "-") for v in ph.values())
        xv = rec.get("x_visibility") or {}
        lag = ";".join(f"{a},{b}" for a, b in (rec.get("video_turn_lag_frames") or {}).values())
        f = lambda v, n=2: "-" if v is None else f"{v:.{n}f}"  # noqa: E731
        print(f"{rec['episode'][-3:]}/{rec['instance'][-1]}/{rec['bot']:8s} {rec['cell']:20s} {str(rec.get('agent')):2s} {str(rec.get('observer_role')):6s} {lens:24s} "
              f"{str(rec.get('recorder_skipped_steps', '-')):>4s} {f(rec.get('yaw_err_max_deg')):>6s} {f(rec.get('action_vs_pose_max_err_rad'), 4):>7s} "
              f"{str(xv.get('x_last_visible_frame')):>5s} {str(xv.get('x_first_revisible_frame')):>4s} {str(xv.get('x_unobserved_frames')):>5s} "
              f"{str(xv.get('gap_phase_x_hidden_from_both')):>7s} {f(xv.get('observed_blocks_iou')):>5s} {lag:>14s} {rec['clean']} {rec['reasons'] or ''}")
    eps = {k: v[0] for k, v in by_ep.items()}
    clean = [v for v in eps.values() if v["episode_clean"]]
    print(f"\nwrote {len(records)} records to {out}; episodes {len(eps)}, clean episodes {len(clean)}")
    print("clean episodes by cell:", dict(sorted(collections.Counter(v["cell"] for v in clean).items())))
    print("clean episodes by group/G:", dict(sorted(collections.Counter(f"{v['group']}/G{v['gap_frames']}" for v in clean).items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

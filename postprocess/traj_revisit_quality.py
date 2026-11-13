#!/usr/bin/env python3
"""Quality manifest and review images for trajRevisitEval (single-agent
revisit: static, yaw out by theta, hold off-screen for H frames, yaw back,
static).

Input: a flattened dataset root holding trajRevisitEval/{trajectories,
episode_info,videos}. Per recording:

  GT pose (trajectory frames, exact):
    * the scene search succeeded and all phases have frames;
    * phase lengths: static_pre >= 28, turn_out / turn_back = theta / 3 deg
      (+-3 frames), hold = H (+-3), static_post >= 30;
    * the bot does not translate (max deviation from the start position) and
      keeps pitch 0;
    * the turn reaches theta (+-1 deg) and ends on the start yaw (+-0.5 deg);
      yaw steps during the turns are uniform (the largest step is reported;
      a doubled step is a frame the recorder skipped).

  Video (pixels, half resolution, HUD rows excluded):
    * still_pre / still_post: the picture does not change inside the two
      static phases (mean abs grey difference between frames 5 and N-5 of
      the phase, and the largest frame-to-frame difference);
    * revisit_shift_px: no residual image shift between the picture before
      the turn and after the return (phase correlation);
    * revisit_diff: the picture after the return equals the picture before
      the turn (mean abs grey difference between a static_pre frame and a
      static_post frame, and the share of pixels differing by > 12 levels);
    * video_leave_frame / video_return_frame: first video frame that differs
      from the start picture / first frame after the turn that matches it
      again (the video can trail the trajectory by a few frames).

  Pair: the two bots of an episode must be >= MIN_BOT_SEPARATION blocks apart
  (players are rendered up to 128 blocks).

Per recording a review strip (before | turned out | after | 4x difference) is
written to <root>/traj_revisit_review/.

Output: <root>/quality_manifest_traj_revisit.json and a summary.
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

TYPE = "trajRevisitEval"
NAME_RE = re.compile(r"_(\d{6})_(\w+?)_instance_(\d{3})$")
SCALE = 0.5
HUD_TOP_FRACTION = 0.80          # rows below this hold the hotbar / hand
STEP_DEG = 3.0
MIN_PRE, MIN_POST = 28, 30
LEN_TOL = 3
MAX_POS_DEV = 0.02               # blocks
MAX_PITCH_DEG = 0.5
THETA_TOL_DEG = 1.0
RETURN_TOL_DEG = 0.5
STILL_MAX = 1.5                  # mean abs grey difference inside static_pre
STILL_POST_MAX = 2.5             # ... inside static_post (the encoder still refines the picture after the turn)
STILL_STEP_MAX = 3.0             # largest frame-to-frame difference inside a static phase
REVISIT_MAX = 4.0                # mean abs grey difference before vs after (clouds drift, see README)
SHIFT_MAX_PX = 1.0               # residual image shift before vs after
MATCH_MAX = 4.0                  # a frame "shows the start picture" below this
MIN_BOT_SEPARATION = 140.0
VIDEO_LAG_MARGIN = 10            # frames skipped at phase edges when sampling the video


def wrap_deg(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def load_video(path: Path) -> tuple[list[np.ndarray], list[np.ndarray]]:
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


def diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    rows = int(a.shape[0] * HUD_TOP_FRACTION)
    return np.abs(a[:rows] - b[:rows])


def analyze(info: dict, frames: list, video: Path, review: Path | None) -> dict:
    reasons: list[str] = []
    rec = {k: info.get(k) for k in ("theta_deg", "signed_theta_deg", "direction", "hold_tier", "hold_frames", "scene_check_passed",
                                    "scene_search_attempts", "start_pos", "start_yaw", "window_start_frame", "window_end_frame")}
    rec["n_frames"] = len(frames)
    rec["reasons"] = reasons
    if not info.get("scene_check_passed"):
        reasons.append("scene search failed")
        return rec
    ph = {p["name"]: p for p in info.get("phases", [])}
    rec["phases"] = {n: [p.get("start_frame"), p.get("end_frame")] for n, p in ph.items()}
    H = info["hold_frames"]
    need = ["static_pre", "turn_out", "turn_back", "static_post"] + (["hold"] if H > 0 else [])
    if any(ph.get(n, {}).get("start_frame") is None for n in need):
        reasons.append("phase frames missing (recording cut short?)")
        return rec
    length = {n: (p["end_frame"] - p["start_frame"] + 1 if p.get("start_frame") is not None else 0) for n, p in ph.items()}
    rec["phase_lengths"] = length
    turn = round(info["theta_deg"] / STEP_DEG)
    if length["static_pre"] < MIN_PRE:
        reasons.append(f"static_pre {length['static_pre']} frames")
    if length["static_post"] < MIN_POST:
        reasons.append(f"static_post {length['static_post']} frames")
    for n in ("turn_out", "turn_back"):
        if abs(length[n] - turn) > LEN_TOL:
            reasons.append(f"{n} {length[n]} frames (expected {turn})")
    if abs(length["hold"] - H) > LEN_TOL:
        reasons.append(f"hold {length['hold']} frames (expected {H})")

    # GT pose over the window
    w0, w1 = ph["static_pre"]["start_frame"], ph["static_post"]["end_frame"]
    win = frames[w0:w1 + 1]
    pos = np.array([[f["x"], f["y"], f["z"]] for f in win])
    yaw = np.degrees(np.unwrap([f["yaw"] for f in win]))
    pitch = np.degrees([f["pitch"] for f in win])
    rec["pos_dev_max"] = float(np.abs(pos - pos[0]).max())
    rec["pitch_abs_max_deg"] = float(np.abs(pitch).max())
    rel = yaw - yaw[0]
    peak = rel[np.argmax(np.abs(rel))]
    rec["turn_reached_deg"] = float(peak)
    rec["return_error_deg"] = float(rel[-1])
    steps = np.abs(np.diff(yaw))
    moving = steps[steps > 0.01]
    rec["turn_frames_gt"] = int(len(moving))
    rec["yaw_step_median_deg"] = float(np.median(moving)) if len(moving) else None
    rec["yaw_step_max_deg"] = float(moving.max()) if len(moving) else None
    rec["recorder_skipped_frames"] = int((moving > 1.5 * STEP_DEG).sum())
    # yaw inside the static phases must not move at all
    pre_rel = rel[: length["static_pre"]]
    post_rel = rel[len(rel) - length["static_post"]:]
    rec["static_yaw_dev_deg"] = float(max(np.abs(pre_rel).max(), np.abs(post_rel - post_rel[-1]).max()))
    if rec["pos_dev_max"] > MAX_POS_DEV:
        reasons.append(f"moved {rec['pos_dev_max']:.3f} blocks")
    if rec["pitch_abs_max_deg"] > MAX_PITCH_DEG:
        reasons.append(f"pitch {rec['pitch_abs_max_deg']:.2f} deg")
    if abs(peak - info["signed_theta_deg"]) > THETA_TOL_DEG:
        reasons.append(f"turned {peak:.1f} deg (planned {info['signed_theta_deg']})")
    if abs(rec["return_error_deg"]) > RETURN_TOL_DEG:
        reasons.append(f"return error {rec['return_error_deg']:.2f} deg")
    if rec["static_yaw_dev_deg"] > 0.1:
        reasons.append(f"yaw moved {rec['static_yaw_dev_deg']:.2f} deg inside a static phase")

    # Video
    grey, colour = load_video(video)
    rec["video_frames"] = len(grey)
    if len(grey) < w1 + 1:
        reasons.append(f"video has {len(grey)} frames, window ends at {w1}")
        return rec
    pre0, pre1 = ph["static_pre"]["start_frame"], ph["static_pre"]["end_frame"]
    post0, post1 = ph["static_post"]["start_frame"], ph["static_post"]["end_frame"]
    ref = grey[pre1 - 3]

    def still(a: int, b: int) -> tuple[float, float]:
        span = float(diff(grey[a], grey[b]).mean())
        step = max(float(diff(grey[i], grey[i + 1]).mean()) for i in range(a, b))
        return span, step

    rec["still_pre"], rec["still_pre_step_max"] = still(pre0 + 2, pre1 - 3)
    rec["still_post"], rec["still_post_step_max"] = still(post0 + VIDEO_LAG_MARGIN, post1 - 2)
    after = grey[post0 + VIDEO_LAG_MARGIN + 2]
    d = diff(ref, after)
    # Residual image shift before vs after (phase correlation, full-resolution
    # pixels): the camera did not come back to the same direction.
    rows = int(ref.shape[0] * HUD_TOP_FRACTION)
    (sx, sy), _ = cv2.phaseCorrelate(ref[:rows], after[:rows])
    rec["revisit_shift_px"] = [float(sx / SCALE), float(sy / SCALE)]
    if max(abs(sx), abs(sy)) / SCALE > SHIFT_MAX_PX:
        reasons.append(f"picture after the return is shifted by ({sx / SCALE:+.1f}, {sy / SCALE:+.1f}) px")
    rec["revisit_diff"] = float(d.mean())
    rec["revisit_changed_pixel_fraction"] = float((d > 12).mean())
    to_ref = [float(diff(ref, g).mean()) for g in grey]
    leave = next((i for i in range(pre1 - 3, len(grey)) if to_ref[i] > MATCH_MAX), None)
    mid = (ph["turn_out"]["end_frame"] + ph["turn_back"]["start_frame"]) // 2
    back = next((i for i in range(mid, len(grey)) if to_ref[i] < MATCH_MAX), None)
    rec["video_leave_frame"] = leave
    rec["video_return_frame"] = back
    rec["video_lag_leave"] = None if leave is None else leave - ph["turn_out"]["start_frame"]
    rec["video_lag_return"] = None if back is None else back - post0
    rec["away_diff"] = to_ref[mid]
    if rec["still_pre"] > STILL_MAX or rec["still_pre_step_max"] > STILL_STEP_MAX:
        reasons.append(f"start picture not still ({rec['still_pre']:.2f}, step {rec['still_pre_step_max']:.2f})")
    if rec["still_post"] > STILL_POST_MAX or rec["still_post_step_max"] > STILL_STEP_MAX:
        reasons.append(f"end picture not still ({rec['still_post']:.2f}, step {rec['still_post_step_max']:.2f})")
    if rec["revisit_diff"] > REVISIT_MAX:
        reasons.append(f"picture after the return differs ({rec['revisit_diff']:.2f})")
    if back is None or back > post0 + VIDEO_LAG_MARGIN:
        reasons.append("video does not show the start picture again in time")

    if review is not None:
        dimg = cv2.cvtColor(np.clip(np.abs(grey[pre1 - 3] - after) * 4, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        tiles = [colour[pre1 - 3].copy(), colour[mid].copy(), colour[post0 + VIDEO_LAG_MARGIN + 2].copy(), dimg]
        labels = [f"before f{pre1 - 3}", f"away f{mid}", f"after f{post0 + VIDEO_LAG_MARGIN + 2}", f"|diff| x4  mean {rec['revisit_diff']:.2f}"]
        for t, lab in zip(tiles, labels):
            cv2.putText(t, lab, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
            cv2.putText(t, lab, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        review.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(review), np.hstack(tiles))
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--no-review", action="store_true")
    args = ap.parse_args()

    base = args.root / TYPE
    review_dir = args.root / "traj_revisit_review"
    records = []
    for info_path in sorted((base / "episode_info").glob("*_episode_info.json")):
        stem = info_path.name[: -len("_episode_info.json")]
        m = NAME_RE.search(stem)
        if not m:
            continue
        ep, bot, inst = m.groups()
        info = json.loads(info_path.read_text())
        traj, video = base / "trajectories" / f"{stem}.json", base / "videos" / f"{stem}.mp4"
        rec = {"type": TYPE, "recording": stem, "episode": ep, "instance": inst, "bot": bot,
               "video": f"{TYPE}/videos/{stem}.mp4"}
        if not traj.exists() or not video.exists():
            rec.update({"reasons": ["trajectory or video missing"], "clean": False})
            records.append(rec)
            continue
        frames = json.loads(traj.read_text())
        tier = info.get("hold_tier")
        review = None if args.no_review else review_dir / f"ep{ep}_inst{inst}_{bot}_{tier}_{info.get('direction')}.jpg"
        rec.update(analyze(info, frames, video, review))
        if review is not None and review.exists():
            rec["review_image"] = str(review.relative_to(args.root))
        records.append(rec)

    # The two bots of an episode must be out of each other's sight.
    by_ep = collections.defaultdict(list)
    for r in records:
        by_ep[(r["instance"], r["episode"])].append(r)
    for pair in by_ep.values():
        pos = [r.get("start_pos") for r in pair]
        if len(pair) == 2 and all(pos):
            d = math.dist(pos[0], pos[1])
            for r in pair:
                r["other_bot_distance"] = d
                if d < MIN_BOT_SEPARATION:
                    r["reasons"].append(f"other bot {d:.0f} blocks away")
    for r in records:
        r["clean"] = not r["reasons"]

    out = args.out or (args.root / "quality_manifest_traj_revisit.json")
    out.write_text(json.dumps(records, indent=1))
    f2 = lambda v: "-" if v is None else f"{v:.2f}"  # noqa: E731
    print(f"{'ep/inst/bot':16s} {'tier':5s} {'dir':5s} {'pre/out/hold/back/post':24s} {'turn':>7s} {'ret':>6s} {'still pre/post':>14s} {'revisit':>7s} {'shift px':>11s} {'lag out/back':>12s} clean")
    for r in records:
        L = r.get("phase_lengths") or {}
        lens = "/".join(str(L.get(n, "-")) for n in ("static_pre", "turn_out", "hold", "turn_back", "static_post"))
        print(f"{r['episode'][-3:]}/{r['instance'][-1]}/{r['bot']:9s} {str(r.get('hold_tier')):5s} {str(r.get('direction')):5s} {lens:24s} "
              f"{f2(r.get('turn_reached_deg')):>7s} {f2(r.get('return_error_deg')):>6s} {f2(r.get('still_pre')) + '/' + f2(r.get('still_post')):>14s} "
              f"{f2(r.get('revisit_diff')):>7s} {'/'.join(f2(v) for v in (r.get('revisit_shift_px') or [None, None])):>11s} {str(r.get('video_lag_leave')) + '/' + str(r.get('video_lag_return')):>12s} {r['clean']} {r['reasons'] or ''}")
    clean = [r for r in records if r["clean"]]
    print(f"\nwrote {len(records)} records to {out}; clean {len(clean)}")
    print("clean by tier/direction:", dict(sorted(collections.Counter(f"{r['hold_tier']}/{r['direction']}" for r in clean).items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

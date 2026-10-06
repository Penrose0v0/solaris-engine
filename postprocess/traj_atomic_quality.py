#!/usr/bin/env python3
"""Per-episode quality manifest for the traj*Eval atomic-trajectory datasets.

Input: a flattened dataset root (see flatten_overhead_dataset.py) with one
subdir per episode type, each containing trajectories/ and episode_info/.

For every recording it reads the commanded parameters from episode_info and
measures what the bot actually did from the per-frame pose stream, then flags
whether the episode is a "clean" single-variable motion:

  trajStaticEval      no motion at all
  trajYawSweepEval    swept amplitude and measured angular rate match the
                      commanded values, no translation, no pitch change
  trajPitchSweepEval  same for pitch, no yaw change
  trajForwardEval     horizontal distance ~ target, no vertical motion, no
                      yaw/pitch change during the drive
  trajOrbitEval       yaw hits target arc, path length ~ r*theta, no vertical
                      motion
  trajSegmentsEval    checked per segment (see analyze_segments): a translation
                      segment moves along the pressed key's direction in the
                      camera frame with no rotation; a rotation segment hits
                      the commanded angle and rate with no translation. Both
                      bots run a script, so both recordings are moving cameras.

In normal (terrain) worlds the translation types can be blocked by trees or
walk off slopes, which is what the flags are for. The moving-camera recording
is the one whose bot_name == sweep_bot; the peer recording is a static camera.

Output: <root>/quality_manifest.json (list of records) and a summary on stdout.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from traj_segments import resolve_segment_frames

TYPES = [
    "trajStaticEval",
    "trajYawSweepEval",
    "trajPitchSweepEval",
    "trajForwardEval",
    "trajOrbitEval",
    "trajSegmentsEval",
]
DEG = 180.0 / math.pi
CAMERA_EPS_RAD = 2e-4


def wrap_deg(d: float) -> float:
    return (d + 180.0) % 360.0 - 180.0


def horiz(a, b) -> float:
    return math.hypot(b["x"] - a["x"], b["z"] - a["z"])


def path_len(frames) -> float:
    return sum(horiz(frames[k], frames[k + 1]) for k in range(len(frames) - 1))


def motion_window(frames, key):
    """Indices of frames where the given action key is active (forward=True
    or camera != [0,0]), restricted to after the sneak marker."""
    sneak = [k for k, f in enumerate(frames) if f["action"].get("sneak")]
    start = sneak[-1] + 1 if sneak else 0
    if key == "forward":
        idx = [k for k in range(start, len(frames)) if frames[k]["action"].get("forward")]
    else:
        # Ignore sub-0.01-deg deltas: a forced look leaves a float-rounding
        # residue that shows up as one extra frame of "motion".
        idx = [
            k for k in range(start, len(frames))
            if max(abs(c) for c in frames[k]["action"].get("camera", [0, 0])) > CAMERA_EPS_RAD
        ]
    return start, idx


def contiguous_runs(idx, max_gap=3):
    """Split sorted frame indices into runs separated by more than max_gap."""
    runs = []
    for k in idx:
        if runs and k - runs[-1][-1] <= max_gap:
            runs[-1].append(k)
        else:
            runs.append([k])
    return runs


MOVE_KEYS = ("forward", "back", "left", "right")
# Camera-frame axis (forward, right) each movement key should move along.
KEY_AXIS = {"forward": (1, 0), "back": (-1, 0), "left": (0, -1), "right": (0, 1)}


def nominal_walk_blocks(hold_ticks: int) -> float:
    """Distance covered on level ground by holding one movement key for
    hold_ticks and then coasting to a stop (vanilla walk physics: per tick
    v <- 0.546 v + 0.098 while the key is down, v <- 0.546 v after)."""
    v = dist = 0.0
    for _ in range(hold_ticks):
        v = v * 0.546 + 0.098
        dist += v
    while v > 0.003:
        v *= 0.546
        dist += v
    return dist


def analyze_segment(seg: dict, frames: list) -> dict:
    """Check one trajSegmentsEval segment against its command. Returns the
    segment with metrics, clean flag and the list of reasons it is not clean.

    Angles follow the mineflayer convention of the pose stream (yaw 0 = north
    = -z, growing counter-clockwise; pitch positive = up). Translation is
    expressed in the camera frame of the segment's first frame: forward,
    right, up."""
    out = {k: seg.get(k) for k in ("idx", "action", "commanded", "start_frame", "end_frame")}
    m, reasons = {}, []
    out["metrics"], out["reasons"] = m, reasons
    s, e = seg.get("start_frame"), seg.get("end_frame")
    if s is None or e is None:
        reasons.append("segment not recorded")
        out["clean"] = False
        return out
    win = frames[s: e + 1]
    a, b = win[0], win[-1]
    m["frames"] = len(win)
    if not 32 <= len(win) <= 38:
        reasons.append(f"segment has {len(win)} frames, expected ~35")

    yaw0 = a["yaw"]
    dx, dz = b["x"] - a["x"], b["z"] - a["z"]
    m["forward"] = dx * -math.sin(yaw0) + dz * -math.cos(yaw0)
    m["right"] = dx * math.cos(yaw0) + dz * -math.sin(yaw0)
    m["up"] = b["y"] - a["y"]
    m["y_range"] = max(f["y"] for f in win) - min(f["y"] for f in win)
    yaw_steps = [wrap_deg((win[k + 1]["yaw"] - win[k]["yaw"]) * DEG) for k in range(len(win) - 1)]
    pitch_steps = [(win[k + 1]["pitch"] - win[k]["pitch"]) * DEG for k in range(len(win) - 1)]
    m["yaw_delta_deg"] = sum(yaw_steps)
    m["pitch_delta_deg"] = sum(pitch_steps)
    m["pitch_start_deg"] = a["pitch"] * DEG
    m["settle_step"] = horiz(win[-2], b) if len(win) > 1 else 0.0

    if any(f.get("isInWater") for f in win):
        reasons.append("in water")
    for key in ("jump", "sprint", "sneak"):
        if any(f["action"].get(key) for f in win):
            reasons.append(f"{key} pressed")

    action = seg["action"]
    cmd = seg["commanded"]
    if action in MOVE_KEYS:
        fa, ra = KEY_AXIS[action]
        along = m["forward"] * fa + m["right"] * ra
        lateral = m["forward"] * ra - m["right"] * fa
        nominal = nominal_walk_blocks(cmd["hold_ticks"])
        m["along"] = along
        m["lateral"] = lateral
        m["nominal_blocks"] = nominal
        m["along_ratio"] = along / nominal
        held = sum(bool(f["action"].get(action)) for f in win)
        m["key_frames"] = held
        others = [k for k in MOVE_KEYS if k != action and any(f["action"].get(k) for f in win)]
        if others:
            reasons.append(f"other movement keys pressed: {','.join(others)}")
        if abs(held - cmd["hold_ticks"]) > 3:
            reasons.append(f"key held {held} frames, commanded {cmd['hold_ticks']}")
        if not 0.9 <= m["along_ratio"] <= 1.1:
            reasons.append(f"moved {along:.2f} blocks along {action}, expected {nominal:.2f}")
        if abs(lateral) > 0.1:
            reasons.append(f"sideways drift {lateral:.2f} blocks")
        if abs(m["up"]) > 0.05 or m["y_range"] > 0.05:
            reasons.append(f"vertical motion (dy={m['up']:.2f}, range={m['y_range']:.2f})")
        if max((abs(v) for v in yaw_steps), default=0) > 0.1 or abs(m["yaw_delta_deg"]) > 0.3:
            reasons.append(f"yaw changed {m['yaw_delta_deg']:.2f} deg")
        if max((abs(v) for v in pitch_steps), default=0) > 0.1 or abs(m["pitch_delta_deg"]) > 0.3:
            reasons.append(f"pitch changed {m['pitch_delta_deg']:.2f} deg")
        if abs(m["pitch_start_deg"]) > 0.5:
            reasons.append(f"camera not level (pitch {m['pitch_start_deg']:.2f} deg)")
        if m["settle_step"] > 0.02:
            reasons.append("still moving at segment end")
    else:
        axis = "yaw" if action.startswith("yaw") else "pitch"
        steps = yaw_steps if axis == "yaw" else pitch_steps
        other = m["pitch_delta_deg"] if axis == "yaw" else m["yaw_delta_deg"]
        other_steps = pitch_steps if axis == "yaw" else yaw_steps
        amp = m[f"{axis}_delta_deg"]
        m["amplitude_err_deg"] = amp - cmd["degrees"]
        # Median non-zero per-frame step x 20 fps (robust to the frame
        # sampler straddling two physics ticks or catching none).
        moving = sorted(abs(v) for v in steps if abs(v) > CAMERA_EPS_RAD * DEG)
        m["rotation_frames"] = len(moving)
        m["omega_measured_deg_per_sec"] = moving[len(moving) // 2] * 20.0 if moving else 0.0
        m["omega_ratio"] = m["omega_measured_deg_per_sec"] / cmd["omega_deg_per_sec"]
        m["xz_drift"] = math.hypot(dx, dz)
        keys = [k for k in MOVE_KEYS if any(f["action"].get(k) for f in win)]
        if keys:
            reasons.append(f"movement keys pressed: {','.join(keys)}")
        if abs(m["amplitude_err_deg"]) > 1.0:
            reasons.append(f"{axis} turned {amp:.2f} deg, commanded {cmd['degrees']}")
        if not 0.9 <= m["omega_ratio"] <= 1.1:
            reasons.append(
                f"{axis} rate {m['omega_measured_deg_per_sec']:.1f} deg/s, commanded {cmd['omega_deg_per_sec']}")
        if max((abs(v) for v in other_steps), default=0) > 0.1 or abs(other) > 0.3:
            reasons.append(f"{'pitch' if axis == 'yaw' else 'yaw'} changed {other:.2f} deg")
        if m["xz_drift"] > 0.05 or abs(m["up"]) > 0.05:
            reasons.append(f"translated (xz={m['xz_drift']:.2f}, dy={m['up']:.2f})")
    out["clean"] = not reasons
    return out


def analyze(etype: str, info: dict, frames: list) -> dict:
    rec = {
        "type": etype,
        "episode": None,
        "bot": info["bot_name"],
        "world_type": info.get("world_type"),
        "moving_camera": etype in ("trajStaticEval", "trajSegmentsEval") or info.get("sweep_bot") == info["bot_name"],
        "params": {k: v for k, v in info.get("eval_metadata", {}).items() if k != "bots_chosen"},
        "num_frames": len(frames),
        "metrics": {},
        "clean": None,
    }
    m = rec["metrics"]
    first, last = frames[0], frames[-1]

    if etype == "trajSegmentsEval":
        if any("start_frame" not in seg for seg in info.get("segments", [])):
            resolve_segment_frames(info, frames)
        rec["scene_check_passed"] = info.get("scene_check_passed")
        rec["segments"] = [analyze_segment(seg, frames) for seg in info.get("segments", [])]
        m["segments_clean"] = sum(s["clean"] for s in rec["segments"])
        m["segments_total"] = len(rec["segments"])
        rec["clean"] = bool(rec["segments"]) and m["segments_clean"] == m["segments_total"]
        return rec

    if etype == "trajStaticEval":
        m["xz_drift"] = horiz(first, last)
        m["dy"] = last["y"] - first["y"]
        m["yaw_drift_deg"] = abs(wrap_deg((last["yaw"] - first["yaw"]) * DEG))
        m["pitch_drift_deg"] = abs((last["pitch"] - first["pitch"]) * DEG)
        rec["clean"] = m["xz_drift"] < 0.05 and abs(m["dy"]) < 0.05 and m["yaw_drift_deg"] < 0.5 and m["pitch_drift_deg"] < 0.5
        return rec

    if not rec["moving_camera"]:
        # Static peer camera: only check it really stayed put.
        m["xz_drift"] = horiz(first, last)
        m["dy"] = last["y"] - first["y"]
        rec["clean"] = m["xz_drift"] < 0.05 and abs(m["dy"]) < 0.05
        return rec

    target = info["sweep_degrees"]
    omega = info["omega_deg_per_sec"]

    if etype in ("trajYawSweepEval", "trajPitchSweepEval"):
        key = "yaw" if etype == "trajYawSweepEval" else "pitch"
        other = "pitch" if key == "yaw" else "yaw"
        start, idx = motion_window(frames, "camera")
        a = frames[start]
        m["xz_drift"] = horiz(a, last)
        m["dy"] = last["y"] - a["y"]
        m[f"{other}_drift_deg"] = abs(wrap_deg((last[other] - a[other]) * DEG))
        if not idx:
            m["error"] = "no camera motion"
            rec["clean"] = False
            return rec
        # The sweep is the last contiguous run of camera motion (the pitch
        # sweep may be preceded by a separate normalize-to-horizon run).
        runs = contiguous_runs(idx)
        run = runs[-1]
        lo, hi = max(run[0] - 1, 0), run[-1]
        # Accumulate wrapped per-frame deltas so +-180 deg sweeps and the
        # 0/2pi seam are handled.
        amp = sum(wrap_deg((frames[k + 1][key] - frames[k][key]) * DEG) for k in range(lo, hi))
        m["camera_motion_runs"] = len(runs)
        m["sweep_start_frame"] = run[0]
        m["sweep_end_frame"] = run[-1]
        m["sweep_frames"] = len(run)
        m["expected_sweep_frames"] = round(abs(target) / omega * 20)
        m["amplitude_deg"] = amp
        m["amplitude_err_deg"] = amp - target
        # Median per-frame step x 20 fps. Robust to the partial last step of
        # sweeps whose duration is not a whole number of ticks and to frames
        # that straddle two physics ticks.
        steps = sorted(abs(wrap_deg((frames[k + 1][key] - frames[k][key]) * DEG)) for k in range(lo, hi))
        m["omega_measured_deg_per_sec"] = steps[len(steps) // 2] * 20.0
        m["omega_mean_deg_per_sec"] = abs(amp) / (len(run) / 20.0)
        m["omega_ratio"] = m["omega_measured_deg_per_sec"] / omega
        m[f"{key}_at_sweep_start_deg"] = frames[lo][key] * DEG
        m[f"{key}_final_deg"] = last[key] * DEG
        rec["clean"] = (
            abs(m["amplitude_err_deg"]) < 1.0
            and 0.85 <= m["omega_ratio"] <= 1.15
            and m[f"{other}_drift_deg"] < 0.5
            and m["xz_drift"] < 0.05
            and abs(m["dy"]) < 0.05
        )
        return rec

    if etype == "trajForwardEval":
        start, idx = motion_window(frames, "forward")
        if not idx:
            rec["clean"] = False
            m["error"] = "no forward frames"
            return rec
        a, b = frames[idx[0]], frames[idx[-1]]
        seg = frames[idx[0]: idx[-1] + 1]
        dist_target = info["distance_blocks"]
        m["dist_target"] = dist_target
        m["dist_achieved"] = horiz(a, b)
        m["dist_ratio"] = m["dist_achieved"] / dist_target
        m["dy"] = b["y"] - a["y"]
        m["y_min_rel"] = min(f["y"] for f in seg) - a["y"]
        m["y_max_rel"] = max(f["y"] for f in seg) - a["y"]
        m["yaw_drift_deg"] = abs(wrap_deg((b["yaw"] - a["yaw"]) * DEG))
        m["pitch_at_drive_deg"] = a["pitch"] * DEG
        # The pitch normalization to the horizon can overlap the first drive
        # frames (recordings made before the settle-wait fix). Report the
        # first drive frame at which pitch has settled; frames from there on
        # are pure translation. None if pitch never settles during the drive.
        settled = [k for k in idx if abs(frames[k]["pitch"]) * DEG < 0.05]
        m["pitch_settled_frame"] = settled[0] if settled else None
        m["pitch_overlap_frames"] = (settled[0] - idx[0]) if settled else len(idx)
        m["pitch_drift_after_settle_deg"] = (
            abs((b["pitch"] - frames[settled[0]]["pitch"]) * DEG) if settled else None
        )
        m["drive_frames"] = len(idx)
        m["drive_start_frame"] = idx[0]
        m["drive_end_frame"] = idx[-1]
        m["expected_drive_frames"] = info.get("traj_expected_ticks")
        m["mean_speed_blocks_per_sec"] = m["dist_achieved"] / (len(idx) / 20.0)
        rec["clean"] = (
            0.9 <= m["dist_ratio"] <= 1.1
            and abs(m["dy"]) < 0.5
            and m["y_min_rel"] > -0.5
            and m["y_max_rel"] < 0.5
            and m["yaw_drift_deg"] < 0.5
            and settled is not None and len(settled) > 0
            and m["pitch_drift_after_settle_deg"] < 0.5
        )
        return rec

    if etype == "trajOrbitEval":
        start, idx = motion_window(frames, "forward")
        if not idx:
            rec["clean"] = False
            m["error"] = "no forward frames"
            return rec
        a, b = frames[idx[0]], frames[idx[-1]]
        seg = frames[idx[0]: idx[-1] + 1]
        r = info["nominal_radius_blocks"]
        theta = abs(target) * math.pi / 180.0
        # The first drive frame already carries one yaw step (omega/20 deg)
        # and the last step lands one tick after the drive's last frame, so
        # measure the yaw delta from the frame before the drive to the end of
        # the episode (the bot is static after stopAll).
        pre = frames[idx[0] - 1] if idx[0] > 0 else a
        m["yaw_delta_deg"] = (last["yaw"] - pre["yaw"]) * DEG
        m["yaw_err_deg"] = wrap_deg(m["yaw_delta_deg"] - target)
        m["omega_measured_deg_per_sec"] = abs(target + m["yaw_err_deg"]) / (len(idx) / 20.0)
        m["pitch_drift_deg"] = abs((b["pitch"] - a["pitch"]) * DEG)
        m["nominal_radius"] = r
        m["path_len_achieved"] = path_len(seg)
        m["path_len_nominal"] = r * theta
        m["path_ratio"] = m["path_len_achieved"] / m["path_len_nominal"]
        m["chord_achieved"] = horiz(a, b)
        m["chord_nominal"] = 2 * r * math.sin(theta / 2)
        m["chord_ratio"] = m["chord_achieved"] / m["chord_nominal"] if m["chord_nominal"] > 1e-6 else None
        m["dy"] = b["y"] - a["y"]
        m["y_min_rel"] = min(f["y"] for f in seg) - a["y"]
        m["y_max_rel"] = max(f["y"] for f in seg) - a["y"]
        m["drive_frames"] = len(idx)
        m["drive_start_frame"] = idx[0]
        m["drive_end_frame"] = idx[-1]
        m["expected_drive_frames"] = info.get("traj_expected_ticks")
        # An unobstructed level orbit reaches ~0.96 of the nominal path length
        # (walk acceleration ramp), so 0.85 separates clean from blocked runs.
        rec["clean"] = (
            abs(m["yaw_err_deg"]) < 1.0
            and m["path_ratio"] > 0.85
            and abs(m["dy"]) < 0.5
            and m["y_min_rel"] > -0.5
            and m["y_max_rel"] < 0.5
            and m["pitch_drift_deg"] < 0.5
        )
        return rec

    raise ValueError(etype)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, type=Path, help="flattened dataset root (one subdir per type)")
    ap.add_argument("--out", type=Path, default=None, help="manifest path (default: <root>/quality_manifest.json)")
    args = ap.parse_args()

    records = []
    for etype in TYPES:
        info_dir = args.root / etype / "episode_info"
        traj_dir = args.root / etype / "trajectories"
        if not info_dir.is_dir():
            continue
        for info_path in sorted(info_dir.glob("*_episode_info.json")):
            base = info_path.name[: -len("_episode_info.json")]
            traj_path = traj_dir / f"{base}.json"
            if not traj_path.exists():
                continue
            info = json.loads(info_path.read_text())
            frames = json.loads(traj_path.read_text())
            rec = analyze(etype, info, frames)
            rec["episode"] = base
            rec["video"] = f"{etype}/videos/{base}.mp4"
            rec["trajectory"] = f"{etype}/trajectories/{base}.json"
            records.append(rec)

    out = args.out or (args.root / "quality_manifest.json")
    out.write_text(json.dumps(records, indent=1))

    print(f"wrote {len(records)} records to {out}\n")
    print(f"{'type':20s} {'recordings':>10s} {'moving':>7s} {'moving&clean':>13s}  clean moving-camera episodes per param combo")
    for etype in TYPES:
        rs = [r for r in records if r["type"] == etype]
        if not rs:
            continue
        mv = [r for r in rs if r["moving_camera"]]
        clean = [r for r in mv if r["clean"]]
        combos = {}
        for r in mv:
            key = json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in sorted(r["params"].items())})
            combos.setdefault(key, [0, 0])
            combos[key][0] += 1
            combos[key][1] += r["clean"]
        per = sorted(c[1] for c in combos.values())
        print(f"{etype:20s} {len(rs):10d} {len(mv):7d} {len(clean):13d}  combos={len(combos)} clean/combo min={per[0]} med={per[len(per)//2]} max={per[-1]}")

    seg_recs = [r for r in records if r["type"] == "trajSegmentsEval"]
    if seg_recs:
        segs = [s for r in seg_recs for s in r["segments"]]
        n_clean = sum(r["clean"] for r in seg_recs)
        print(f"\ntrajSegmentsEval per segment: recordings with all segments clean {n_clean}/{len(seg_recs)}, "
              f"segments clean {sum(s['clean'] for s in segs)}/{len(segs)}")
        for action in list(MOVE_KEYS) + ["yaw+", "yaw-", "pitch+", "pitch-"]:
            sub = [s for s in segs if s["action"] == action]
            print(f"  {action:8s} clean {sum(s['clean'] for s in sub):3d}/{len(sub):3d}")
        why = {}
        for s in segs:
            for reason in s["reasons"]:
                # Group by the reason's wording without its numbers.
                key = " ".join(w for w in reason.split() if not any(c.isdigit() for c in w))
                why[key] = why.get(key, 0) + 1
        for key, n in sorted(why.items(), key=lambda kv: -kv[1]):
            print(f"  not clean: {n:3d} x {key}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

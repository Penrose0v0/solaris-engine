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

TYPES = [
    "trajStaticEval",
    "trajYawSweepEval",
    "trajPitchSweepEval",
    "trajForwardEval",
    "trajOrbitEval",
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


def analyze(etype: str, info: dict, frames: list) -> dict:
    rec = {
        "type": etype,
        "episode": None,
        "bot": info["bot_name"],
        "world_type": info.get("world_type"),
        "moving_camera": etype == "trajStaticEval" or info.get("sweep_bot") == info["bot_name"],
        "params": {k: v for k, v in info.get("eval_metadata", {}).items() if k != "bots_chosen"},
        "num_frames": len(frames),
        "metrics": {},
        "clean": None,
    }
    m = rec["metrics"]
    first, last = frames[0], frames[-1]

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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

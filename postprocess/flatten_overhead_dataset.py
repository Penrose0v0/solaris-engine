#!/usr/bin/env python3
"""Flatten a collection batch into a solaris-new2-style dataset directory.

Input layout (produced by orchestrate.py start + postprocess):
    <source-dir>/aligned/<ts>_<ep>_<Bot>_instance_<n>_camera.mp4
    <source-dir>/aligned/<ts>_<ep>_<Bot>_instance_<n>_overhead_camera.mp4   (optional)
    <source-dir>/output/<ts>_<ep>_<Bot>_instance_<n>.json
    <source-dir>/output/<ts2>_<ep>_<Bot>_instance_<n>_episode_info.json

Output layout (grouped by file type):
    <dest>/videos/<batch>_<ep>_<Bot>_instance_<n>.mp4
    <dest>/videos_overhead/<batch>_<ep>_<Bot>_instance_<n>_overhead.mp4     (if present)
    <dest>/trajectories/<batch>_<ep>_<Bot>_instance_<n>.json
    <dest>/episode_info/<batch>_<ep>_<Bot>_instance_<n>_episode_info.json
    <dest>/first_frames/<batch>_<ep>_<Bot>_instance_<n>_first_frame.png
    <dest>/first_frames/<batch>_<ep>_<Bot>_instance_<n>_overhead_first_frame.png

The per-frame JSONs land in trajectories/ because each entry is a full step
record — pose (x/y/z/yaw/pitch), keyboard/mouse action, inventory — not just
the action.

Pass --flat to reproduce the legacy solaris-new2 layout (everything in one
directory) instead.

Episodes whose side-video frame count does not match the action count are
skipped with a warning (same validity rule as prepare_train_dataset.py).
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import cv2

from traj_segments import resolve_event_frames, resolve_segment_frames

TIMESTAMP_PREFIX_RE = re.compile(r"^\d{8}_\d{6}_")


def strip_timestamp(name: str) -> str:
    return TIMESTAMP_PREFIX_RE.sub("", name, count=1)


def frame_count(video: Path) -> int:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError(f"Failed to open video {video}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def action_count(actions_json: Path) -> int:
    with actions_json.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError(f"{actions_json} is not a JSON list")
    return len(data)


def save_first_frame(video: Path, dest: Path) -> bool:
    cap = cv2.VideoCapture(str(video))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return False
    return bool(cv2.imwrite(str(dest), frame))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, type=Path,
                        help="Batch dir containing aligned/ and output/")
    parser.add_argument("--destination-dir", required=True, type=Path)
    parser.add_argument("--batch-name", default="batch_0",
                        help="Prefix for destination filenames (default: batch_0)")
    parser.add_argument("--flat", action="store_true",
                        help="Put every file directly in the destination dir "
                        "(legacy solaris-new2 layout) instead of grouping "
                        "into videos/, trajectories/, episode_info/, "
                        "first_frames/")
    args = parser.parse_args()

    aligned = args.source_dir / "aligned"
    outputs = args.source_dir / "output"
    if not aligned.is_dir() or not outputs.is_dir():
        print(f"aligned/ or output/ missing under {args.source_dir}", file=sys.stderr)
        return 1

    dest = args.destination_dir
    if args.flat:
        videos_dir = overhead_dir = trajectories_dir = info_dir = frames_dir = dest
    else:
        videos_dir = dest / "videos"
        overhead_dir = dest / "videos_overhead"
        trajectories_dir = dest / "trajectories"
        info_dir = dest / "episode_info"
        frames_dir = dest / "first_frames"
    # videos_overhead/ is created lazily so overhead-less runs don't get an
    # empty directory
    for d in {videos_dir, trajectories_dir, info_dir, frames_dir}:
        d.mkdir(parents=True, exist_ok=True)

    # Map stripped episode_info names -> path (episode_info files carry a
    # different timestamp prefix than the action files, so match by suffix).
    episode_infos = {
        strip_timestamp(p.name): p
        for p in outputs.glob("*_episode_info.json")
    }

    copied = skipped = 0
    for side_video in sorted(aligned.glob("*_camera.mp4")):
        if side_video.name.endswith("_overhead_camera.mp4"):
            continue

        stem = side_video.name[: -len("_camera.mp4")]
        actions_json = outputs / f"{stem}.json"
        if not actions_json.exists():
            print(f"SKIP {stem}: action json missing", file=sys.stderr)
            skipped += 1
            continue

        try:
            frames = frame_count(side_video)
            actions = action_count(actions_json)
        except ValueError as exc:
            print(f"SKIP {stem}: {exc}", file=sys.stderr)
            skipped += 1
            continue
        if frames != actions:
            print(
                f"SKIP {stem}: video has {frames} frames but json has {actions} actions",
                file=sys.stderr,
            )
            skipped += 1
            continue

        info_json = episode_infos.get(f"{strip_timestamp(stem)}_episode_info.json")
        if info_json is None:
            print(f"SKIP {stem}: episode_info json missing", file=sys.stderr)
            skipped += 1
            continue

        base = f"{args.batch_name}_{strip_timestamp(stem)}"

        shutil.copy2(side_video, videos_dir / f"{base}.mp4")
        shutil.copy2(actions_json, trajectories_dir / f"{base}.json")
        info = json.loads(info_json.read_text())
        frames_json = json.loads(actions_json.read_text())
        resolved = resolve_segment_frames(info, frames_json)  # trajSegmentsEval
        resolved = resolve_event_frames(info, frames_json) or resolved  # worldModEval
        if resolved:
            # add per-segment start_frame/end_frame or per-event frame_start/frame_end
            (info_dir / f"{base}_episode_info.json").write_text(json.dumps(info, indent=2))
        else:
            shutil.copy2(info_json, info_dir / f"{base}_episode_info.json")
        if not save_first_frame(side_video, frames_dir / f"{base}_first_frame.png"):
            print(f"WARN {stem}: failed to extract first frame", file=sys.stderr)

        overhead_video = aligned / f"{stem}_overhead_camera.mp4"
        if overhead_video.exists():
            overhead_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(overhead_video, overhead_dir / f"{base}_overhead.mp4")
            if not save_first_frame(
                overhead_video, frames_dir / f"{base}_overhead_first_frame.png"
            ):
                print(f"WARN {stem}: failed to extract overhead first frame", file=sys.stderr)

        copied += 1

    print(f"Flattened {copied} episodes to {args.destination_dir} ({skipped} skipped)")
    return 0 if copied > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Frame windows for trajSegmentsEval segments.

The controller cannot see recording frame numbers, so each segment in
episode_info carries its wall-clock window (start_time_ms / end_time_ms, the
same clock as the per-frame renderTime). This maps the window onto the frames
of the recording: start_frame / end_frame are inclusive indices into the
trajectory (and video) of the same recording.
"""
from __future__ import annotations


def resolve_segment_frames(info: dict, frames: list) -> bool:
    """Fill start_frame/end_frame of every segment in ``info`` in place.

    A frame belongs to a segment when start_time_ms <= renderTime <
    end_time_ms. Segments that never ran (missing window) or that no frame
    falls into get None. Returns True if ``info`` has segments.
    """
    segments = info.get("segments")
    if not segments:
        return False
    times = [f["renderTime"] for f in frames]
    for seg in segments:
        t0, t1 = seg.get("start_time_ms"), seg.get("end_time_ms")
        idx = [k for k, t in enumerate(times) if t0 <= t < t1] if t0 is not None and t1 is not None else []
        seg["start_frame"] = idx[0] if idx else None
        seg["end_frame"] = idx[-1] if idx else None
    return True

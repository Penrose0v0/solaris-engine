"""Frame windows for trajSegmentsEval segments and worldModEval events.

The controller cannot see recording frame numbers, so each segment / event in
episode_info carries its wall-clock window (start_time_ms / end_time_ms, the
same clock as the per-frame renderTime). This maps the window onto the frames
of the recording: start_frame / end_frame (segments) and frame_start /
frame_end (events) are inclusive indices into the trajectory (and video) of
the same recording.
"""
from __future__ import annotations


def resolve_event_frames(info: dict, frames: list) -> bool:
    """worldModEval: fill frame_start/frame_end of every event and
    window_start_frame/window_end_frame in ``info`` in place.

    frame_start is the first frame rendered at or after the action started,
    frame_end the last frame rendered before the client saw the world change
    (never earlier than frame_start). Returns True if ``info`` has events.
    """
    events = info.get("events")
    if not events:
        return False
    times = [f["renderTime"] for f in frames]
    for ev in events:
        t0, t1 = ev.get("start_time_ms"), ev.get("end_time_ms")
        if t0 is None or t1 is None:
            ev["frame_start"] = ev["frame_end"] = None
            continue
        after = [k for k, t in enumerate(times) if t >= t0]
        before = [k for k, t in enumerate(times) if t <= t1]
        if not after:
            ev["frame_start"] = ev["frame_end"] = None
            continue
        ev["frame_start"] = after[0]
        ev["frame_end"] = max(after[0], before[-1]) if before else after[0]
    w0, w1 = info.get("window_start_time_ms"), info.get("window_end_time_ms")
    if w0 is not None and w1 is not None:
        idx = [k for k, t in enumerate(times) if w0 <= t < w1]
        info["window_start_frame"] = idx[0] if idx else None
        info["window_end_frame"] = idx[-1] if idx else None
    return True


def resolve_window_frames(info: dict, frames: list) -> bool:
    """coObserveEval (and any episode with only a motion window): fill
    window_start_frame/window_end_frame from window_start_time_ms/
    window_end_time_ms when they are not set yet. Returns True if it did."""
    w0, w1 = info.get("window_start_time_ms"), info.get("window_end_time_ms")
    if w0 is None or w1 is None or "window_start_frame" in info:
        return False
    idx = [k for k, f in enumerate(frames) if w0 <= f["renderTime"] < w1]
    info["window_start_frame"] = idx[0] if idx else None
    info["window_end_frame"] = idx[-1] if idx else None
    return True


def resolve_change_frames(info: dict, frames: list) -> bool:
    """sharedArenaChangeEval: fill change.frame (first frame rendered at or
    after this bot's client received the block change) and the hold frames
    change.hold_start_frame / change.hold_end_frame (from the window ticks).
    Trajectory frames; the video can trail by a few frames (see the quality
    manifest's video_change_frame). Returns True if ``info`` has a change."""
    ch = info.get("change")
    if not ch:
        return False
    t = ch.get("seen_time_ms")
    after = [k for k, f in enumerate(frames) if t is not None and f["renderTime"] >= t]
    ch["frame"] = after[0] if after else None
    w0 = info.get("window_start_time_ms")
    for name, tick in (("hold_start_frame", ch.get("hold_start_tick")), ("hold_end_frame", ch.get("hold_end_tick"))):
        idx = [k for k, f in enumerate(frames) if w0 is not None and tick is not None and f["renderTime"] >= w0 + 50 * tick]
        ch[name] = idx[0] if idx else None
    return True


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


def resolve_phase_frames(info: dict, frames: list) -> bool:
    """trajRevisitEval: fill start_frame/end_frame of every phase in ``info``
    in place (same rule as segments: start_time_ms <= renderTime <
    end_time_ms; a zero-length phase, e.g. the hold of the H=0 tier, gets
    None). Returns True if ``info`` has phases."""
    phases = info.get("phases")
    if not phases:
        return False
    times = [f["renderTime"] for f in frames]
    for ph in phases:
        t0, t1 = ph.get("start_time_ms"), ph.get("end_time_ms")
        idx = [k for k, t in enumerate(times) if t0 <= t < t1] if t0 is not None and t1 is not None else []
        ph["start_frame"] = idx[0] if idx else None
        ph["end_frame"] = idx[-1] if idx else None
    return True

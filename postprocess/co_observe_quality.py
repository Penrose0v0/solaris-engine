#!/usr/bin/env python3
"""Per-episode quality manifest and review montages for the coObserveEval
dataset (shared-world consistency: both bots circle-strafe around a static
concrete landmark).

Input: a flattened dataset root (see flatten_overhead_dataset.py) holding
coObserveEval/{trajectories,episode_info,videos}. The two recordings of an
episode are paired by episode number and instance.

Per episode it pairs the frames of the two recordings by wall clock
(renderTime, nearest frame within PAIR_TOLERANCE_MS) inside the motion window
and computes for every pair

  view_angle_deg   angle between the two camera forward directions (GT
                   yaw/pitch; both bots aim at the landmark, so this is also
                   the angle between the two lines of sight)
  dist_a / dist_b  horizontal distance of each bot to the landmark centre
  landmark_overlap fraction of sample points on the landmark surface and the
                   ground disk around it that project inside BOTH frames
  view_iou         frustum overlap of the two cameras: points of a 3D grid
                   around the arena (+-VOL_HALF blocks, several heights)
                   visible in both views / visible in either view

and per recording

  stuck_frames     frames inside the window (after the first 10) whose
                   horizontal displacement is < STUCK_EPS blocks
  r_err_max        largest |distance to centre - radius_target|
  y_range          vertical travel (steps / falls)
  fov              focal length estimated from the landmark silhouette
                   (the GT box projected with a grid of focal lengths; the
                   one whose outline separates concrete from background
                   best), nominal 514 px (70 deg) vs boosted 452 px (77 deg)
  overlay          HUD overlay heuristics: toast panel (light grey, top
                   right), yellow chat text and chat darkening (bottom left)

Output: <root>/quality_manifest_co_observe.json, one record per episode, a
summary on stdout and <root>/co_observe_review/<ep>_<inst>.png side-by-side
frames (A | B) at three times of the window.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import cv2
import numpy as np

from world_mod_quality import CONCRETE_HUE, EYE_HEIGHT, VFOV_DEG, Video, camera_axes

TYPE = "coObserveEval"
PAIR_TOLERANCE_MS = 30
MIN_WINDOW_FRAMES = 150
OVERLAP_MIN = 0.5
IOU_MIN = 0.5
VOL_HALF = 16
VOL_STEP = 2.0
VOL_HEIGHTS = (0.5, 2.0, 4.0, 7.0)
STUCK_EPS = 0.02
STUCK_SKIP = 10
MAX_STUCK_FRAMES = 3
MAX_R_ERR = 0.75
MAX_Y_RANGE = 0.1
ANGLE_BINS = [0, 15, 30, 45, 60, 75, 90, 120, 180]
F_NOMINAL = 514.2
F_BOOSTED = 452.6
F_GRID = np.arange(400.0, 621.0, 4.0)
FOV_FRAMES = 12
FOV_TOL = 25.0
DEG = 180.0 / math.pi

STEM_RE = re.compile(r"^(?P<prefix>.+?)_(?P<ep>\d{6})_(?P<bot>[A-Za-z]+)_instance_(?P<inst>\d{3})$")


def nominal_focal(height: int) -> float:
    return (height / 2.0) / math.tan(math.radians(VFOV_DEG) / 2.0)


def project_many(pose: dict, pts: np.ndarray, width: int, height: int, focal: float | None = None):
    """Pixel coordinates (N,) x/y and an in-front mask for an (N,3) array."""
    f, r, u = camera_axes(pose["yaw"], pose["pitch"])
    v = pts - np.array([pose["x"], pose["y"] + EYE_HEIGHT, pose["z"]])
    zc = v @ np.array(f)
    xc = v @ np.array(r)
    yc = v @ np.array(u)
    focal = focal or nominal_focal(height)
    with np.errstate(divide="ignore", invalid="ignore"):
        px = width / 2.0 + xc / zc * focal
        py = height / 2.0 - yc / zc * focal
    return px, py, zc > 1e-6


def visible_many(pose, pts, width, height):
    px, py, ok = project_many(pose, pts, width, height)
    return ok & (px >= 0) & (px < width) & (py >= 0) & (py < height)


def forward(yaw: float, pitch: float):
    return camera_axes(yaw, pitch)[0]


def angle_between(f, g) -> float:
    d = max(-1.0, min(1.0, sum(a * b for a, b in zip(f, g))))
    return math.acos(d) * DEG


def landmark_blocks(info: dict) -> np.ndarray:
    """(N,3) min corners of the landmark blocks (block coordinates)."""
    lm = info["landmark"]
    if lm.get("blocks"):
        return np.array([(b["x"], b["y"], b["z"]) for b in lm["blocks"]], dtype=float)
    # older layout: full box between footprint_min and footprint_max
    x0, y0, z0 = lm["footprint_min"]
    x1, y1, z1 = lm["footprint_max"]
    return np.array([(x, y, z) for x in range(x0, x1 + 1) for y in range(y0, y1 + 1) for z in range(z0, z1 + 1)], dtype=float)


def landmark_block_names(info: dict) -> list:
    lm = info["landmark"]
    if lm.get("blocks"):
        return [b["block"] for b in lm["blocks"]]
    return [c["block"] for c in lm.get("columns", [])]


def landmark_points(info: dict) -> np.ndarray:
    """Face centres of every landmark block (all six faces, so the far side
    counts too), plus a ground ring around the structure (radius 2-4)."""
    pts = []
    for x, y, z in landmark_blocks(info):
        pts += [(x + 0.5, y + 0.5, z), (x + 0.5, y + 0.5, z + 1), (x, y + 0.5, z + 0.5), (x + 1, y + 0.5, z + 0.5),
                (x + 0.5, y + 1, z + 0.5), (x + 0.5, y, z + 0.5)]
    cx, y0, cz = info["centre"]
    for r in np.arange(2.0, 4.01, 0.5):
        for a in np.arange(0, 2 * math.pi, math.pi / 6):
            pts.append((cx + r * math.cos(a), y0, cz + r * math.sin(a)))
    return np.array(pts, dtype=float)


def volume_points(centre) -> np.ndarray:
    xs = np.arange(-VOL_HALF, VOL_HALF + 1e-6, VOL_STEP)
    pts = [(centre[0] + dx, centre[1] + h, centre[2] + dz) for dx in xs for dz in xs for h in VOL_HEIGHTS]
    return np.array(pts, dtype=float)


def landmark_corners(info: dict) -> np.ndarray:
    """(N,8,3) corners of every landmark block."""
    out = []
    for x, y, z in landmark_blocks(info):
        out.append([(x + dx, y + dy, z + dz) for dx in (0, 1) for dy in (0, 1) for dz in (0, 1)])
    return np.array(out, dtype=float)


def landmark_silhouette(pose: dict, corners: np.ndarray, width: int, height: int, focal: float):
    """Union of the projected blocks (uint8 mask), or None when a block is
    behind the camera."""
    mask = np.zeros((height, width), np.uint8)
    flat = corners.reshape(-1, 3)
    px, py, ok = project_many(pose, flat, width, height, focal)
    if not ok.all():
        return None
    px = px.reshape(-1, 8)
    py = py.reshape(-1, 8)
    for i in range(px.shape[0]):
        hull = cv2.convexHull(np.stack([px[i], py[i]], 1).astype(np.float32)).astype(np.int32)
        cv2.fillConvexPoly(mask, hull, 1)
    return mask


def window_of(info: dict, frames: list):
    w0, w1 = info.get("window_start_frame"), info.get("window_end_frame")
    if w0 is None or w1 is None:
        t0, t1 = info.get("window_start_time_ms"), info.get("window_end_time_ms")
        if t0 is None or t1 is None:
            return None
        idx = [k for k, f in enumerate(frames) if t0 <= f["renderTime"] < t1]
        if not idx:
            return None
        w0, w1 = idx[0], idx[-1]
    return w0, w1


def recording_motion(info: dict, frames: list, win) -> dict:
    centre = info["centre"]
    r0 = info.get("radius_target")
    out = {"bot": info["bot_name"], "num_frames": len(frames), "window": list(win) if win else None,
           "window_frames": (win[1] - win[0] + 1) if win else None, "radius_target": r0}
    if not win:
        return out
    seg = frames[win[0]: win[1] + 1]
    rs = [math.hypot(f["x"] - centre[0], f["z"] - centre[2]) for f in seg]
    ys = [f["y"] for f in seg]
    stuck = [win[0] + k for k in range(STUCK_SKIP, len(seg))
             if math.hypot(seg[k]["x"] - seg[k - 1]["x"], seg[k]["z"] - seg[k - 1]["z"]) < STUCK_EPS]
    keys = sorted({k for f in seg for k in ("forward", "back", "left", "right", "jump", "sneak", "sprint") if f["action"].get(k)})
    out.update({
        "r_min": min(rs), "r_max": max(rs), "r_mean": float(np.mean(rs)),
        "r_err_max": max(abs(r - r0) for r in rs) if r0 is not None else None,
        "r_final_err": (rs[-1] - r0) if r0 is not None else None,
        "y_range": max(ys) - min(ys),
        "stuck_frames": stuck,
        "keys_pressed": keys,
        "forward_frames": sum(1 for f in seg if f["action"].get("forward")),
        "back_frames": sum(1 for f in seg if f["action"].get("back")),
        "held_items": sorted({(f.get("inventory") or {}).get("hotbar", [{}])[0].get("name", "") for f in seg if False}),
    })
    return out


# ---------------------------------------------------------------------------
# HUD overlays
# ---------------------------------------------------------------------------

def yellow_text_rows(img, roi) -> dict:
    """Yellow chat text (join/leave messages) in the ROI: a thin band of
    saturated yellow pixels spanning a wide x range."""
    hsv = cv2.cvtColor(img[roi], cv2.COLOR_BGR2HSV)
    m = (hsv[..., 0] >= 22) & (hsv[..., 0] <= 38) & (hsv[..., 1] > 120) & (hsv[..., 2] > 200)
    rows = m.sum(axis=1)
    good = np.where(rows >= 25)[0]
    if len(good) < 4:
        return {"hit": False, "pixels": int(m.sum())}
    # longest run of rows (allowing 2-row gaps) with text-like width
    best = 0
    run_rows = []
    cur = [good[0]]
    for a, b in zip(good[:-1], good[1:]):
        if b - a <= 3:
            cur.append(b)
        else:
            if len(cur) > best:
                best, run_rows = len(cur), cur
            cur = [b]
    if len(cur) > best:
        best, run_rows = len(cur), cur
    if best < 4 or best > 40:
        return {"hit": False, "pixels": int(m.sum())}
    cols = np.where(m[run_rows].any(axis=0))[0]
    width = int(cols[-1] - cols[0]) if len(cols) else 0
    return {"hit": width >= 120, "pixels": int(m.sum()), "rows": best, "width": width}


def overlay_scan(video: Video) -> dict:
    """Heuristic HUD overlay detection over the whole video: toast panel
    (light grey, top-right), yellow chat text and chat darkening (bottom
    left). Returns frame indices per kind."""
    if not video.n:
        return {"toast_frames": [], "chat_text_frames": [], "chat_dark_frames": [], "checked": 0}
    W, H = video.width, video.height
    toast_roi = (slice(0, 96), slice(W - 480, W))
    chat_roi = (slice(H - 200, H - 80), slice(0, 760))
    chat_ref = (slice(H - 200, H - 80), slice(760, W))
    toast_frac, chat_mean, ref_mean, text_hits = [], [], [], []
    video.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    k = 0
    while True:
        ok, img = video.cap.read()
        if not ok:
            break
        hsv = cv2.cvtColor(img[toast_roi], cv2.COLOR_BGR2HSV)
        grey = (hsv[..., 1] < 40) & (hsv[..., 2] > 150) & (hsv[..., 2] < 235)
        toast_frac.append(float(grey.mean()))
        chat_mean.append(float(img[chat_roi].mean()))
        ref_mean.append(float(img[chat_ref].mean()))
        if yellow_text_rows(img, chat_roi)["hit"]:
            text_hits.append(k)
        k += 1
    video.cache.clear()
    tf = np.array(toast_frac)
    cm = np.array(chat_mean)
    rm = np.array(ref_mean)
    toast = [int(i) for i in np.where((tf > 0.35) & (np.median(tf) < 0.15))[0]]
    ratio = cm / np.maximum(np.median(cm), 1.0)
    ref_ratio = rm / np.maximum(np.median(rm), 1.0)
    dark = [int(i) for i in np.where((ratio < 0.6) & (ref_ratio > 0.8))[0]]
    return {"toast_frames": toast, "chat_text_frames": text_hits, "chat_dark_frames": dark, "checked": len(tf)}


# ---------------------------------------------------------------------------
# FOV from the landmark silhouette
# ---------------------------------------------------------------------------

def concrete_mask(img, blocks: list, exclude: set):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    m = np.zeros(h.shape, bool)
    for b in blocks:
        color = b.replace("_concrete", "")
        if color in exclude:
            continue
        if color == "white":
            m |= (s < 30) & (v > 185)
        elif color == "black":
            m |= v < 40
        elif color in CONCRETE_HUE:
            for lo, hi in CONCRETE_HUE[color]:
                m |= (h >= lo) & (h <= hi) & (s > 110) & (v > 45)
    return m


def hull_mask(shape, pts):
    m = np.zeros(shape, np.uint8)
    hull = cv2.convexHull(pts.astype(np.float32)).astype(np.int32)
    cv2.fillConvexPoly(m, hull, 1)
    return m


def landmark_focal(video: Video, frames: list, info: dict, win) -> dict:
    """Median over FOV_FRAMES frames of the focal length whose projected
    landmark outline best separates concrete from background."""
    if not video.n or not win:
        return {"f_est": None, "verdict": "unknown", "n_frames": 0}
    corners = landmark_corners(info)
    blocks = landmark_block_names(info)
    colors = {b.replace("_concrete", "") for b in blocks}
    W, H = video.width, video.height
    # Static frames only (lead before the window, tail after the bots have
    # stopped): while the bots move, the rendered view trails the GT pose by
    # a few frames and the silhouette no longer matches the projection.
    def static_at(k):
        if k < 2 or k + 2 >= len(frames):
            return False
        a, b = frames[k - 2], frames[k + 2]
        return (abs(a["x"] - b["x"]) < 1e-3 and abs(a["z"] - b["z"]) < 1e-3
                and abs(a["yaw"] - b["yaw"]) < 1e-3 and abs(a["pitch"] - b["pitch"]) < 1e-3)
    # The lead frames (before the motion) and the tail frames (after it)
    # are estimated separately: before the motion the camera client has not
    # always settled on the GT pose yet, so the tail decides when it has
    # enough frames.
    lead = [k for k in range(len(frames)) if k < win[0] - 2 and static_at(k)]
    tail = [k for k in range(len(frames)) if k > win[1] + 12 and static_at(k)]
    if not lead and not tail:
        return {"f_est": None, "verdict": "unknown", "n_frames": 0}
    pick = lambda c: sorted({int(c[i]) for i in np.linspace(0, len(c) - 1, FOV_FRAMES).astype(int)}) if c else []  # noqa: E731
    ks = pick(lead) + pick(tail)
    tail_set = set(pick(tail))
    ests = []
    ests_tail = []
    k_small = np.ones((5, 5), np.uint8)
    k_big = np.ones((17, 17), np.uint8)
    k_er = np.ones((7, 7), np.uint8)
    for k in ks:
        img = video.frame(int(k))
        if img is None:
            continue
        pose = frames[int(k)]
        px, py, ok = project_many(pose, corners.reshape(-1, 3), W, H, F_NOMINAL)
        if not ok.all() or px.min() < 20 or px.max() > W - 20 or py.min() < 20 or py.max() > H - 160:
            continue
        base = landmark_silhouette(pose, corners, W, H, F_NOMINAL)
        # Only the left/right edges at mid-height are scored: above the
        # landmark is sky (matches blue/cyan), below it its shadow.
        x0, x1 = px.min(), px.max()
        y0, y1 = py.min(), py.max()
        roi = np.zeros((H, W), bool)
        ya, yb = int(y0 + 0.15 * (y1 - y0)), int(y1 - 0.15 * (y1 - y0))
        roi[ya:yb, : int(x0 + 0.3 * (x1 - x0))] = True
        roi[ya:yb, int(x1 - 0.3 * (x1 - x0)):] = True
        # Colours that also occur in the background beside the landmark are
        # not used (lime vs grass, yellow/orange vs sand, blue vs sky).
        ring = ((cv2.dilate(base, np.ones((61, 61), np.uint8)) - cv2.dilate(base, k_big)).astype(bool)) & roi
        exclude = set()
        for color in colors:
            m = concrete_mask(img, [f"{color}_concrete"], set())
            if ring.sum() and m[ring].mean() > 0.08:
                exclude.add(color)
        if len(colors - exclude) == 0:
            continue
        cm = concrete_mask(img, blocks, exclude).astype(np.float32)
        best = None
        for f in F_GRID:
            m = landmark_silhouette(pose, corners, W, H, float(f))
            if m is None:
                continue
            inner = cv2.erode(m, k_er).astype(bool) & roi
            outer = (cv2.dilate(m, k_big) - cv2.dilate(m, k_small)).astype(bool) & roi
            if inner.sum() < 50 or outer.sum() < 50:
                continue
            score = float(cm[inner].mean() - cm[outer].mean())
            if best is None or score > best[0]:
                best = (score, float(f))
        if best is not None and best[0] > 0.3:
            ests.append(best[1])
            if int(k) in tail_set:
                ests_tail.append(best[1])
    n_tail = len(ests_tail)
    ests_lead = ests[: len(ests) - n_tail]
    if len(ests) < 3:
        return {"f_est": None, "verdict": "unknown", "n_frames": len(ests)}
    f_est = float(np.median(ests))
    f_lead = float(np.median(ests_lead)) if len(ests_lead) >= 3 else None
    f_tail = float(np.median(ests_tail)) if n_tail >= 3 else None
    # A real boosted camera shows in both the lead and the tail; a single
    # segment reading low is usually the partner bot in front of the
    # landmark or the camera not yet settled before the motion.
    parts = [f for f in (f_lead, f_tail) if f is not None] or [f_est]
    if all(abs(f - F_NOMINAL) <= FOV_TOL for f in parts):
        verdict = "nominal"
    elif all(abs(f - F_BOOSTED) <= FOV_TOL for f in parts):
        verdict = "boosted"
    else:
        verdict = "unclear"
    return {"f_est": f_est, "fov_est_deg": 2 * math.degrees(math.atan((H / 2.0) / f_est)), "verdict": verdict,
            "n_frames": len(ests), "f_lead": f_lead, "f_tail": f_tail, "estimates": ests}


# ---------------------------------------------------------------------------
# Episode analysis
# ---------------------------------------------------------------------------

def pair_frames(fa: list, fb: list, wa, wb):
    """(ka, kb) pairs: every A frame in A's window matched to the nearest B
    frame inside B's window by renderTime."""
    tb = np.array([f["renderTime"] for f in fb[wb[0]: wb[1] + 1]])
    pairs = []
    for ka in range(wa[0], wa[1] + 1):
        ta = fa[ka]["renderTime"]
        j = int(np.argmin(np.abs(tb - ta)))
        if abs(tb[j] - ta) <= PAIR_TOLERANCE_MS:
            pairs.append((ka, wb[0] + j))
    return pairs


def analyze_episode(a, b) -> dict:
    (base_a, info_a, fr_a, vid_a), (base_b, info_b, fr_b, vid_b) = a, b
    rec = {
        "type": TYPE,
        "bots": [info_a["bot_name"], info_b["bot_name"]],
        "primary": info_a.get("primary"),
        "world_type": info_a.get("world_type"),
        "scene_check_passed": bool(info_a.get("scene_check_passed")) and bool(info_b.get("scene_check_passed")),
        "direction_mode": info_a.get("direction_mode"),
        "initial_view_angle_deg": info_a.get("initial_view_angle_deg"),
        "centre": info_a.get("centre"),
        "landmark": info_a.get("landmark"),
        "terraform": (info_a.get("terraform") or info_b.get("terraform")),
        "scene_search_attempts": info_a.get("scene_search_attempts") or info_b.get("scene_search_attempts"),
        "encountered_error": bool(info_a.get("encountered_error") or info_b.get("encountered_error")),
        "reasons": [],
    }
    if not rec["scene_check_passed"]:
        rec["reasons"].append("scene search failed")
        rec["clean"] = False
        return rec
    if rec["encountered_error"]:
        rec["reasons"].append("controller error")
    va, vb = Video(vid_a), Video(vid_b)
    wa, wb = window_of(info_a, fr_a), window_of(info_b, fr_b)
    rec["recordings"] = {}
    for info, frames, win, vid in ((info_a, fr_a, wa, va), (info_b, fr_b, wb, vb)):
        m = recording_motion(info, frames, win)
        m["overlay"] = overlay_scan(vid)
        m["fov"] = landmark_focal(vid, frames, info, win)
        m["strafe_key"] = info.get("strafe_key")
        m["start_bearing_deg"] = info.get("start_bearing_deg")
        rec["recordings"][info["bot_name"]] = m
        tag = info["bot_name"]
        if not win or m["window_frames"] < MIN_WINDOW_FRAMES:
            rec["reasons"].append(f"{tag}: window {m['window_frames']} frames < {MIN_WINDOW_FRAMES}")
            continue
        if len(m["stuck_frames"]) > MAX_STUCK_FRAMES:
            rec["reasons"].append(f"{tag}: stuck {len(m['stuck_frames'])} frames")
        if m["r_err_max"] is not None and m["r_err_max"] > MAX_R_ERR:
            rec["reasons"].append(f"{tag}: radius error {m['r_err_max']:.2f} > {MAX_R_ERR}")
        if m["y_range"] > MAX_Y_RANGE:
            rec["reasons"].append(f"{tag}: vertical travel {m['y_range']:.2f}")
        if m["fov"]["verdict"] == "boosted":
            rec["reasons"].append(f"{tag}: camera rendered at boosted FOV (f {m['fov']['f_est']:.0f})")
        ov = m["overlay"]
        in_win = lambda ks: [k for k in ks if win[0] <= k <= win[1]]  # noqa: E731
        if in_win(ov["toast_frames"]):
            rec["reasons"].append(f"{tag}: toast suspected on {len(in_win(ov['toast_frames']))} frames")
        # chat_dark_frames (darkening of the chat area) is informational only:
        # terrain entering the corner trips it too.
        if in_win(ov["chat_text_frames"]):
            rec["reasons"].append(f"{tag}: chat text suspected on {len(in_win(ov['chat_text_frames']))} frames")
    pairs = pair_frames(fr_a, fr_b, wa, wb) if wa and wb else []
    rec["num_pairs"] = len(pairs)
    if wa and wb and len(pairs) < MIN_WINDOW_FRAMES:
        rec["reasons"].append(f"only {len(pairs)} paired frames")
    lm_pts = landmark_points(info_a)
    vol_pts = volume_points(info_a["centre"])
    centre = info_a["centre"]
    aim = np.array([info_a.get("aim_point") or [centre[0], centre[1] + 1.0, centre[2]]], dtype=float)
    per = []
    for ka, kb in pairs:
        pa, pb = fr_a[ka], fr_b[kb]
        la = visible_many(pa, lm_pts, va.width, va.height)
        lb = visible_many(pb, lm_pts, vb.width, vb.height)
        ga = visible_many(pa, vol_pts, va.width, va.height)
        gb = visible_many(pb, vol_pts, vb.width, vb.height)
        union = int((ga | gb).sum())
        per.append({
            "ka": ka, "kb": kb, "dt_ms": pa["renderTime"] - pb["renderTime"],
            "view_angle_deg": angle_between(forward(pa["yaw"], pa["pitch"]), forward(pb["yaw"], pb["pitch"])),
            "dist_a": math.hypot(pa["x"] - centre[0], pa["z"] - centre[2]),
            "dist_b": math.hypot(pb["x"] - centre[0], pb["z"] - centre[2]),
            "landmark_overlap": float((la & lb).mean()),
            "view_iou": float((ga & gb).sum() / union) if union else 0.0,
            "aim_in_a": bool(visible_many(pa, aim, va.width, va.height)[0]),
            "aim_in_b": bool(visible_many(pb, aim, vb.width, vb.height)[0]),
        })
    rec["pairs"] = per
    if per:
        angs = np.array([p["view_angle_deg"] for p in per])
        ovs = np.array([p["landmark_overlap"] for p in per])
        ious = np.array([p["view_iou"] for p in per])
        rec["view_angle"] = {"first": float(angs[0]), "last": float(angs[-1]), "min": float(angs.min()),
                             "max": float(angs.max()), "median": float(np.median(angs)),
                             "share_30_90": float(((angs >= 30) & (angs <= 90)).mean()),
                             "share_below_30": float((angs < 30).mean()), "share_above_90": float((angs > 90).mean())}
        rec["landmark_overlap_share"] = float((ovs >= OVERLAP_MIN).mean())
        rec["view_iou"] = {"min": float(ious.min()), "median": float(np.median(ious)), "max": float(ious.max()),
                           "share_ge_0.5": float((ious >= 0.5).mean()), "share_ge_0.3": float((ious >= 0.3).mean())}
        rec["aim_in_both_share"] = float(np.mean([p["aim_in_a"] and p["aim_in_b"] for p in per]))
        best = run = 0
        for o in ious >= IOU_MIN:
            run = run + 1 if o else 0
            best = max(best, run)
        rec["iou_longest_run"] = int(best)
    va.close()
    vb.close()
    rec["clean"] = not rec["reasons"]
    return rec


def montage(rec: dict, a, b, out_path: Path):
    (base_a, info_a, fr_a, vid_a), (base_b, info_b, fr_b, vid_b) = a, b
    pairs = rec.get("pairs") or []
    if not pairs:
        return False
    va, vb = Video(vid_a), Video(vid_b)
    picks = [pairs[min(5, len(pairs) - 1)], pairs[len(pairs) // 2], pairs[max(0, len(pairs) - 6)]]
    rows = []
    for p in picks:
        ia, ib = va.frame(p["ka"]), vb.frame(p["kb"])
        if ia is None or ib is None:
            continue
        ia = cv2.resize(ia, (640, 360))
        ib = cv2.resize(ib, (640, 360))
        row = np.concatenate([ia, ib], axis=1)
        txt = (f"{info_a['bot_name']} f{p['ka']} r={p['dist_a']:.1f} | {info_b['bot_name']} f{p['kb']} r={p['dist_b']:.1f} | "
               f"angle {p['view_angle_deg']:.0f} deg  view IoU {p['view_iou']:.2f}  mode {rec['direction_mode']}")
        cv2.putText(row, txt, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(row, txt, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        rows.append(row)
    va.close()
    vb.close()
    if not rows:
        return False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(out_path), np.concatenate(rows, axis=0)))


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


def pct(vals, q):
    return float(np.percentile(vals, q)) if len(vals) else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, type=Path, help="flattened dataset root (one subdir per type)")
    ap.add_argument("--out", type=Path, default=None, help="manifest path (default: <root>/quality_manifest_co_observe.json)")
    ap.add_argument("--no-montage", action="store_true")
    args = ap.parse_args()

    records = []
    review_dir = args.root / "co_observe_review"
    for (inst, ep), bots in sorted(load(args.root).items()):
        names = sorted(bots)
        if len(names) < 2:
            print(f"episode {ep} instance {inst}: only {names}, skipped")
            records.append({"type": TYPE, "episode": ep, "instance": inst, "bots": names, "clean": False,
                            "scene_check_passed": bool(bots[names[0]][1].get("scene_check_passed")),
                            "direction_mode": bots[names[0]][1].get("direction_mode"),
                            "reasons": ["second recording missing"]})
            continue
        a, b = bots[names[0]], bots[names[1]]
        rec = analyze_episode(a, b)
        rec["episode"] = ep
        rec["instance"] = inst
        rec["recording_files"] = {name: f"{TYPE}/videos/{v[0]}.mp4" for name, v in bots.items()}
        if not args.no_montage:
            path = review_dir / f"ep{ep}_inst{inst}.png"
            if montage(rec, a, b, path):
                rec["montage"] = str(path.relative_to(args.root))
        records.append(rec)
        print(f"episode {ep} instance {inst}: mode={rec.get('direction_mode')} clean={rec['clean']} {rec['reasons']}")

    out = args.out or (args.root / "quality_manifest_co_observe.json")
    out.write_text(json.dumps(records, indent=1))
    print(f"\nwrote {len(records)} episode records to {out}\n")

    n = len(records)
    passed = [r for r in records if r.get("scene_check_passed")]
    clean = [r for r in records if r.get("clean")]
    print(f"episodes {n}, scene_check_passed {len(passed)}, clean {len(clean)}, failure rate {(n - len(clean)) / max(n, 1):.0%}")
    why = {}
    for r in records:
        for reason in r["reasons"]:
            key = " ".join(w for w in reason.split() if not any(c.isdigit() for c in w))
            why[key] = why.get(key, 0) + 1
    for key, cnt in sorted(why.items(), key=lambda kv: -kv[1]):
        print(f"  not clean: {cnt:3d} x {key}")

    print(f"\n{'ep/inst':8s} {'mode':9s} {'theta0':>6s} {'rA':>5s} {'rB':>5s} {'win':>7s} {'pairs':>5s} {'ang first/med/last':>19s} "
          f"{'in30-90':>7s} {'IoU med':>7s} {'IoU>.5':>6s} {'rerr':>9s} {'stuck':>5s} {'fov A/B':>9s} {'overlay':>7s}")
    for r in records:
        if not r.get("recordings"):
            print(f"{r['episode'][-3:]}/{r['instance'][-1]:4s} {str(r.get('direction_mode')):9s} scene failed after {r.get('scene_search_attempts')} attempts")
            continue
        recs = list(r["recordings"].values())
        va = r.get("view_angle") or {}
        iou = r.get("view_iou") or {}
        fov = "/".join((rc["fov"]["verdict"][:4] if rc["fov"]["f_est"] is None else f"{rc['fov']['f_est']:.0f}") for rc in recs)
        ovl = sum(len(rc["overlay"]["toast_frames"]) + len(rc["overlay"]["chat_text_frames"]) + len(rc["overlay"]["chat_dark_frames"]) for rc in recs)
        print(f"{r['episode'][-3:]}/{r['instance'][-1]:4s} {r['direction_mode']:9s} {r['initial_view_angle_deg']:6.1f} "
              f"{recs[0]['radius_target']:5.1f} {recs[1]['radius_target']:5.1f} "
              f"{str(recs[0]['window_frames']) + '/' + str(recs[1]['window_frames']):>7s} {r.get('num_pairs', 0):5d} "
              f"{va.get('first', float('nan')):6.0f}/{va.get('median', float('nan')):5.0f}/{va.get('last', float('nan')):5.0f} "
              f"{va.get('share_30_90', float('nan')):7.0%} {iou.get('median', float('nan')):7.2f} {iou.get('share_ge_0.5', float('nan')):6.0%} "
              f"{(recs[0].get('r_err_max') or 0):4.2f}/{(recs[1].get('r_err_max') or 0):4.2f} "
              f"{len(recs[0].get('stuck_frames', [])):2d}/{len(recs[1].get('stuck_frames', [])):2d} "
              f"{fov:>9s} {ovl:7d}")

    for mode in ("same", "opposite", None):
        sub = [r for r in passed if r.get("pairs") and (mode is None or r["direction_mode"] == mode)]
        if not sub:
            continue
        angs = np.concatenate([[p["view_angle_deg"] for p in r["pairs"]] for r in sub])
        dists = np.concatenate([[p["dist_a"] for p in r["pairs"]] + [p["dist_b"] for p in r["pairs"]] for r in sub])
        ovs = np.concatenate([[p["landmark_overlap"] for p in r["pairs"]] for r in sub])
        ious = np.concatenate([[p["view_iou"] for p in r["pairs"]] for r in sub])
        hist, _ = np.histogram(angs, bins=ANGLE_BINS)
        label = mode or "all"
        print(f"\n[{label}] {len(sub)} episodes, {len(angs)} frame pairs")
        print(f"  view angle deg: p5 {pct(angs, 5):.0f} p25 {pct(angs, 25):.0f} med {pct(angs, 50):.0f} p75 {pct(angs, 75):.0f} p95 {pct(angs, 95):.0f}; "
              f"share <30: {(angs < 30).mean():.0%}, 30-90: {((angs >= 30) & (angs <= 90)).mean():.0%}, >90: {(angs > 90).mean():.0%}")
        print("  histogram " + " ".join(f"[{ANGLE_BINS[i]}-{ANGLE_BINS[i+1]}):{hist[i]/len(angs):.0%}" for i in range(len(hist))))
        print(f"  distance to centre: p5 {pct(dists, 5):.1f} med {pct(dists, 50):.1f} p95 {pct(dists, 95):.1f}")
        print(f"  landmark in both views (>= {OVERLAP_MIN} of its sample points): {(ovs >= OVERLAP_MIN).mean():.0%} of pairs")
        print(f"  view IoU (3D grid +-{VOL_HALF} blocks): p5 {pct(ious, 5):.2f} med {pct(ious, 50):.2f} p95 {pct(ious, 95):.2f}; "
              f"share >= 0.5: {(ious >= 0.5).mean():.0%}, >= 0.3: {(ious >= 0.3).mean():.0%}")
    fov = {}
    for r in passed:
        for rc in (r.get("recordings") or {}).values():
            v = rc["fov"]["verdict"]
            fov[v] = fov.get(v, 0) + 1
    print(f"\ncamera FOV from the landmark silhouette (recordings): {fov}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

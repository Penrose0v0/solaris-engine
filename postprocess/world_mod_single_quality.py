#!/usr/bin/env python3
"""Per-episode quality manifest and review montages for the worldModSingleEval
dataset (one placement or one break per episode, item held in the actor's
main hand throughout).

Re-uses the worldModEval checks of world_mod_quality.py (world change
confirmed by the server, observer client saw it, both recordings static
before / during / after the event, target inside the frame and line of
sight clear, rendered change detected, block-silhouette FOV check, window
105-129 frames) and adds

  post_event_frames   frames from the end of the event to the window end
                      (>= MIN_POST_FRAMES in both recordings)
  held_item           the actor's main-hand item is the expected one on
                      every frame of the window (concrete for a placement,
                      the pickaxe for a break) and the off-hand is empty;
                      the quick-bar slot never changes
  overlay             HUD overlay heuristics of co_observe_quality (toast
                      panel, yellow chat text) inside the window

Output: <root>/quality_manifest_world_mod_single.json, a summary on stdout
and <root>/world_mod_single_review/<ep>_<inst>.png (actor | observer before
the event, after it and at the window end).
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import cv2
import numpy as np

import world_mod_quality as wmq
from co_observe_quality import overlay_scan

TYPE = "worldModSingleEval"
wmq.TYPE = TYPE
MIN_POST_FRAMES = 80


def hand_check(info: dict, frames: list, win) -> dict:
    expected = info.get("held_item")
    if not win or win[0] is None or win[1] is None:
        return {"expected": expected, "ok": None}
    seg = frames[win[0]: win[1] + 1]

    def item_name(f, key):
        it = (f.get("inventory") or {}).get(key)
        return it.get("name") if isinstance(it, dict) else None

    held = collections.Counter(item_name(f, "heldItem") for f in seg)
    off = collections.Counter(item_name(f, "offHandItem") for f in seg)
    slots = collections.Counter((f.get("inventory") or {}).get("quickBarSlot") for f in seg)
    held_ok = set(held) == {expected}
    off_ok = set(off) == {None}
    return {"expected": expected, "held": dict(held), "offhand": dict(off), "quick_bar_slots": dict(slots),
            "ok": bool(held_ok and off_ok and len(slots) == 1), "held_ok": held_ok, "offhand_ok": off_ok, "slot_ok": len(slots) == 1}


def montage(rec: dict, actor, observer, out_path: Path):
    (base_a, info_a, fr_a, vid_a), (base_o, info_o, fr_o, vid_o) = actor, observer
    ev_a = (info_a.get("events") or [None])[0]
    ev_o = (info_o.get("events") or [None])[0]
    if not ev_a or ev_a.get("frame_start") is None or not ev_o or ev_o.get("frame_start") is None:
        return False
    va, vo = wmq.Video(vid_a), wmq.Video(vid_o)
    wa = (info_a.get("window_start_frame"), info_a.get("window_end_frame"))
    wo = (info_o.get("window_start_frame"), info_o.get("window_end_frame"))
    picks = [
        ("before", ev_a["frame_start"] - 6, ev_o["frame_start"] - 6),
        ("after", ev_a["frame_end"] + 12, ev_o["frame_end"] + 12),
        ("window end", (wa[1] or 0) - 2, (wo[1] or 0) - 2),
    ]
    rows = []
    for label, ka, ko in picks:
        ia, io = va.frame(ka), vo.frame(ko)
        if ia is None or io is None:
            continue
        row = np.concatenate([cv2.resize(ia, (640, 360)), cv2.resize(io, (640, 360))], axis=1)
        txt = f"{label} | {rec['interaction']} {rec['block']} | actor {info_a['bot_name']} f{ka} | observer {info_o['bot_name']} f{ko} | angle {rec.get('view_angle_deg', 0):.0f} deg"
        cv2.putText(row, txt, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(row, txt, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        rows.append(row)
    va.close()
    vo.close()
    if not rows:
        return False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(out_path), np.concatenate(rows, axis=0)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--no-montage", action="store_true")
    args = ap.parse_args()

    records = []
    review_dir = args.root / "world_mod_single_review"
    for (inst, ep), bots in sorted(wmq.load(args.root).items()):
        actor = next((v for v in bots.values() if v[1].get("role") == "actor"), None)
        observer = next((v for v in bots.values() if v[1].get("role") == "observer"), None)
        if actor is None:
            print(f"episode {ep} instance {inst}: no actor recording, skipped")
            continue
        av = wmq.Video(actor[3])
        ov = wmq.Video(observer[3]) if observer else wmq.Video(None)
        rec = wmq.analyze_episode(actor[1], actor[2], av, observer[1] if observer else None, observer[2] if observer else None, ov)
        rec["type"] = TYPE
        rec["episode"] = ep
        rec["instance"] = inst
        rec["interaction"] = actor[1].get("interaction")
        rec["block"] = actor[1].get("block")
        rec["target_cell"] = actor[1].get("target_cell")
        rec["recordings"] = {name: f"{TYPE}/videos/{v[0]}.mp4" for name, v in bots.items()}
        # Extra checks.
        for side, item in (("actor", actor), ("observer", observer)):
            if item is None:
                continue
            info, frames = item[1], item[2]
            ev = (info.get("events") or [{}])[0]
            w1 = info.get("window_end_frame")
            post = (w1 - ev["frame_end"]) if w1 is not None and ev.get("frame_end") is not None else None
            rec[f"{side}_post_event_frames"] = post
            if post is None or post < MIN_POST_FRAMES:
                rec["reasons"].append(f"{side}: {post} frames after the event < {MIN_POST_FRAMES}")
            win = (info.get("window_start_frame"), info.get("window_end_frame"))
            ov_ = overlay_scan(wmq.Video(item[3]))
            in_win = lambda ks: [k for k in ks if win[0] is not None and win[0] <= k <= win[1]]  # noqa: E731
            rec[f"{side}_overlay"] = {"toast_frames": in_win(ov_["toast_frames"]), "chat_text_frames": in_win(ov_["chat_text_frames"])}
            # The chat-text heuristic (yellow text band bottom left) fires on
            # the yellow concrete swinging through that corner during the
            # placement and on sunflowers (prod001: 4 false positives, no
            # real chat; the cameras run with chat hidden), so it stays
            # informational; the toast heuristic remains a reason.
            if rec[f"{side}_overlay"]["toast_frames"]:
                rec["reasons"].append(f"{side}: toast suspected on {len(rec[f'{side}_overlay']['toast_frames'])} frames")
        hc = hand_check(actor[1], actor[2], (actor[1].get("window_start_frame"), actor[1].get("window_end_frame")))
        rec["hand_check"] = hc
        if hc["ok"] is False:
            rec["reasons"].append("actor hand/hotbar not as expected")
        rec["clean"] = bool(rec.get("events")) and not rec["reasons"] and all(e["clean"] for e in rec["events"])
        av.close()
        ov.close()
        if not args.no_montage and observer is not None:
            path = review_dir / f"ep{ep}_inst{inst}.png"
            if montage(rec, actor, observer, path):
                rec["montage"] = str(path.relative_to(args.root))
        records.append(rec)
        print(f"episode {ep} instance {inst}: {rec['interaction']} {rec['block']} clean={rec['clean']} {rec['reasons']}")

    out = args.out or (args.root / "quality_manifest_world_mod_single.json")
    out.write_text(json.dumps(records, indent=1))
    print(f"\nwrote {len(records)} episode records to {out}\n")

    n = len(records)
    clean = [r for r in records if r["clean"]]
    print(f"episodes {n}, scene_check_passed {sum(r['scene_check_passed'] for r in records)}, clean {len(clean)}, failure rate {(n - len(clean)) / max(n, 1):.0%}")
    why = collections.Counter()
    for r in records:
        for reason in r["reasons"]:
            why[" ".join(w for w in reason.split() if not any(c.isdigit() for c in w))] += 1
        for e in r.get("events", []):
            for reason in e["reasons"]:
                why["event: " + " ".join(w for w in reason.split() if not any(c.isdigit() for c in w))] += 1
    for key, cnt in why.most_common():
        print(f"  not clean: {cnt:3d} x {key}")
    print(f"\n{'ep/inst':8s} {'type':6s} {'block':18s} {'band':6s} {'ang':>4s} {'win':>4s} {'pre':>4s} {'post A/O':>8s} {'changed':>7s} {'obs_seen':>8s} {'vis A/O':>7s} {'pix A/O':>7s} {'fov A/O':>9s} {'hand':>5s} {'ovl':>4s}")
    for r in records:
        if not r.get("events"):
            print(f"{r['episode'][-3:]}/{r['instance'][-1]:4s} scene failed")
            continue
        e = r["events"][0]
        ra, ro = r.get("actor_recording") or {}, r.get("observer_recording") or {}
        ea = (ra.get("events") or [{}])[0]
        band = "-".join(str(b) for b in (r.get("view_angle_band") or [])) or "-"
        fov = "/".join((f"{x['fov_scale']:.2f}" if x and x.get("fov_scale") is not None else "n/a") for x in (ra, ro))
        ovl = sum(len(r.get(f"{s}_overlay", {}).get(k, [])) for s in ("actor", "observer") for k in ("toast_frames", "chat_text_frames"))
        print(f"{r['episode'][-3:]}/{r['instance'][-1]:4s} {r['interaction']:6s} {str(r['block']):18s} {band:6s} {r.get('view_angle_deg', 0):4.0f} "
              f"{str(r.get('window_frames')):>4s} {str(ea.get('frame_start')):>4s} {str(r.get('actor_post_event_frames')) + '/' + str(r.get('observer_post_event_frames')):>8s} "
              f"{str(e.get('world_changed')):>7s} {str(e.get('observer_seen')):>8s} {str(e.get('actor_visible'))[:1] + '/' + str(e.get('observer_visible'))[:1]:>7s} "
              f"{str(e.get('actor_pixel_visible'))[:1] + '/' + str(e.get('observer_pixel_visible'))[:1]:>7s} {fov:>9s} {str(r['hand_check'].get('ok'))[:1]:>5s} {ovl:4d}")
    for t in ("place", "break"):
        sub = [r for r in records if r.get("interaction") == t]
        if sub:
            posts = [r.get("actor_post_event_frames") for r in sub if r.get("actor_post_event_frames") is not None]
            print(f"[{t}] episodes {len(sub)}, clean {sum(r['clean'] for r in sub)}, frames after the event (actor): "
                  f"{'min %d med %d max %d' % (min(posts), sorted(posts)[len(posts)//2], max(posts)) if posts else 'n/a'}")
    hands = collections.Counter((r["hand_check"].get("expected"), tuple(sorted(str(k) for k in (r["hand_check"].get("held") or {})))) for r in records if r.get("events"))
    print("held item (expected -> seen):", dict(hands))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

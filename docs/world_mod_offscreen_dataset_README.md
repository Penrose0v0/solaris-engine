# worldModRevisitEval (EP) and worldModOffscreenEval (DSP)

Two episode types built on the worldModSingleEval interaction: one placement or
one break per episode, the item in the actor's main hand for the whole episode
(concrete block for a placement, diamond pickaxe for a break), off-hand empty,
hotbar slot constant. 20 fps, pitch 0 throughout, no bot ever translates.
Natural worlds, daytime, weather cleared, mob spawning off and entities removed.

- `worldModRevisitEval/` — **EP**, long-term interaction fidelity, single agent:
  interact, look away for H frames, look back.
- `worldModOffscreenEval/` — **DSP**, off-screen state propagation, two agents:
  A interacts while B looks away; after H frames B turns onto the target.

Scene screening: a scene is only used when every resting view (EP: start and
turned-out view; DSP: A's view, B's start and end view) is free of near
occluders: at most 25% of a grid of rays over the frame may end on a solid
block (trunk, leaves, a bank) within 6 blocks (`near_occluder_fraction`).

## Target

Because the pitch stays 0, the target cell sits at eye level: a one-block
pedestal (`pedestal_block`, smooth stone) stands on the ground at the actor's
feet level, 2.8-3.7 blocks straight ahead, and the **target cell** is the cell
on top of it (`target_cell`, feet y + 1; the eye is at y + 1.62).

- place: the target cell is air; the actor puts its concrete block onto the
  pedestal (the crosshair rests on the target cell, not on the pedestal top).
- break: the target cell holds a concrete block (`block`); the actor digs it
  (7 ticks). Block drops are off; break particles are visible.

`yaw` in the trajectories is the mineflayer yaw in radians: 0 = north = -z,
growing counter-clockwise, so "left" turns have growing yaw.

## EP script (frames)

| phase | frames | |
|---|---|---|
| `static_pre` | 10 | still, facing the target |
| `interact` | 4 (place) / 10 (break) | the click is issued on the first frame |
| `static_mid` | 15 | still |
| `turn_out` | 40 | 120 deg at 3 deg per frame |
| `hold` | H = 0 / 40 / 160 | still, target off-screen |
| `turn_back` | 40 | back to the start yaw |
| `static_post` | 40 | still |

Both bots of a world record their own, independent episode about 400 blocks
apart (never in each other's sight); every recording is one episode.
Variants: `k = (3 * instance + episode) % 6` gives H (`k % 3`) and place /
break; the two bots of an episode turn in opposite directions.

## DSP script (frames)

| phase | frames | |
|---|---|---|
| `static_pre` | 10 | nobody moves |
| `interact` | 4 (place) / 10 (break) | A interacts, B does not move |
| `wait` | H = 0 / 40 / 160 | nobody moves; A keeps facing the target |
| `b_turn` | 30 | B turns 90 deg onto the target, 3 deg per frame |
| `static_post` | 30 | nobody moves |

Layout: A (actor) 2.8-3.7 blocks from the target, facing it. B 3.5-6 blocks
from the target, starting with its heading 90 deg off the direction to the
target (the target is 90 deg off B's view centre; half the horizontal FOV is
51.2 deg). A stands on the side B turns towards, so at the start B sees neither
the target nor A; after the turn the target is at B's view centre and A stands
18-42 deg beside it. B is at least 64 deg off A's view centre (never in A's
picture). B's hands are empty. See `layout` in episode_info.

Variants: `k = (6 * instance + episode) % 12` gives H (`k % 3`), place / break
(`floor(k / 3) % 2`) and B's turn direction (`floor(k / 6)`). Which bot
(Alpha / Bravo) is A is drawn per episode (`agent`, `agents`).

## Frames, poses and actions

The script is stepped by the frame recorder: every recorded frame advances it
by one step and the look is set right before the recorder reads the pose. So

- `phases[*].start_frame / end_frame` are exact indices into the recording's
  own trajectory and video;
- `action.camera` of frame i is `[yaw(i) - yaw(i-1), pitch(i) - pitch(i-1)]`
  (radians) on every frame; a frame the recorder skips would show up as one
  frame with a doubled step in both pose and action (flagged in the manifest);
- the interaction appears in the action stream as `place_block` (one frame) or
  `mine` (the frames of the dig), inside `interact`;
- a static lead (camera sync) precedes the window and a static tail of 15
  frames follows it: use `window_start_frame .. window_end_frame`;
- the two recordings of a DSP episode have their own frame indices; the two
  frame grids are offset by less than one frame.

The video comes from a separate camera client and can trail the trajectory by
a few frames (`video_effect_lag_frames` in the quality manifest).

## episode_info

- `eval_kind` (EP / DSP), `hold_frames` (H), `interaction` (place / break),
  `direction` (EP: the bot's turn out; DSP: B's turn), `block`, `target_cell`,
  `pedestal_cell`, `held_item`, `start_pos`, `start_yaw`; DSP also `agent`,
  `agents`, `layout`.
- `phases`: the tables above with `start_frame`, `end_frame`.
- `event`: `trigger_frame` (the click), `action_frames`, **`effect_frame`**
  (first frame rendered at or after the server confirmed the new block state;
  in B's file the same instant on B's frames), `video_effect_frame` (actor recordings: first
  video frame on which the target box is closer to the end picture than to the
  start picture; `video_first_change_frame` is the first visible change, which
  for a break is the start of the dig), and the
  verification flags `server_block_before_ok`, `world_changed`,
  `server_block_end_ok`.
- `target_visibility`, from the GT frustum of every trajectory frame (vertical
  FOV 70 deg, 16:9, eye 1.62 above the feet; the cell is visible when one of
  its 8 corners or its centre is inside the frame; no occlusion test, the
  line of sight is checked at setup): `visible_intervals`,
  `fully_visible_intervals`; EP: `hidden_interval`, `hidden_frames`; DSP:
  `b_visible_intervals`, `b_hidden_interval`, `b_first_visible_frame`,
  `b_effect_frame`, `offscreen_frames_after_effect`,
  `hidden_from_b_until_turn`, `hidden_from_b_at_effect`,
  `target_offset_from_b_view_centre_end_deg`,
  `actor_body_in_b_view_intervals` (all `b_*` frames index B's recording).

## Quality manifest

`quality_manifest_world_mod_offscreen.json`
(postprocess/world_mod_offscreen_quality.py): one record per recording with
`clean`, `episode_clean` (DSP: both recordings clean), `reasons`, `cell`
(kind / interaction / H / direction) and the measured values. Review images
are in `world_mod_offscreen_review/` (columns: static_pre, after the effect,
turned out / before B's turn, static_post, 4x start-vs-end difference).

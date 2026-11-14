# sharedPersistEval: shared-world persistence with a conditioning frame

Two bots stand in a walled courtyard (the crossRevisitEval / sharedArenaEval
courtyard: 15 x 15 blocks inside, walls of 3-wide modules 4-6 blocks tall, no
roof). Neither bot ever translates; pitch is 0 throughout. The first bot looks
at the north wall `x` from the very first frame of the window (the
conditioning frame) and turns away; after a gap in which nobody sees `x`, the
other bot turns to `x`.

## Geometry

Azimuths: north 0, east 90 (clockwise). In the trajectories `yaw` is the
mineflayer yaw in radians, `yaw = -azimuth`; north is `-z`, east is `+x`.

- A stands west, B east, 6 / 6.5 / 7 blocks apart on an east-west line,
  2 or 3 blocks south of the courtyard centre (9.5 or 10.5 blocks from the
  inner face of the north wall).
- **B first** (both turn left): B starts at azimuth 0 (sees `x`, not A),
  A at 130 (sees B 40 deg off its view centre, not `x`). B turns left to 230,
  later A turns left to 0.
- **A first** (both turn right): A starts at 0, B at 230 (sees A, not `x`).
  A turns right to 130, later B turns right to 360.

Horizontal FOV 102.4 deg. Facing north, each bot has the whole north wall
inside its frame. Which bot (Alpha / Bravo) is A is drawn per episode: see
`agent` and `shared_persist.agents`.

## Script (frames at 20 fps)

| phase | frames | |
|---|---|---|
| `static_pre` | 10 | nobody moves; its first frame is the conditioning frame |
| `first_leave` | 33 | the first bot turns 130 deg away: 32 frames of 4 deg, then one of 2 deg |
| `gap` | G = 5 or 55 | nobody moves, nobody sees `x` |
| `second_turn` | 33 | the second bot turns 130 deg to north |
| `second_observe` | 15 | the second bot looks at `x` |

The window is 91 + G frames (`window_start_frame .. window_end_frame`);
`condition_frame` = `window_start_frame`. A static lead precedes the window
(it contains a short camera sync wiggle about 2 s before the window, so do
not condition on frame 0 of the file) and a static tail of 15 frames follows.
The frame of a turn phase with index i shows the pose after i + 1 steps.

Variants, by `k = (2 * instance + episode) % 4`: G = 5 / 55 (bit 0),
B first / A first (bit 1).

## Frames, poses and actions

As in crossRevisitEval the script is driven by the frame recorder: every
recorded frame advances the script by one step and the look is set right
before the recorder reads the pose. Hence

- `phases[*].start_frame / end_frame` are exact indices into the recording's
  own trajectory and video;
- `action.camera` of frame i is `[yaw(i) - yaw(i-1), pitch(i) - pitch(i-1)]`
  in radians, for every frame, one action per frame;
- the two recordings of an episode start at different times, so the same
  moment has different frame indices in the two; each episode_info gives all
  frames in its own recording's indices. The bots' frame grids are offset by
  less than one frame (0-50 ms).

The video comes from a separate camera client and can be off from the
trajectory by a frame or two (`video_turn_lag_frames` in the quality
manifest).

## episode_info

- `shared_persist`: `role_order`, `first_observer_agent`, `first_observer`,
  `second_observer` (bot names), `gap_frames`, `turn_direction`, `agents`,
  start / end azimuths, `bot_distance`, `north_wall_distance`, `x`.
- `agent` (A / B) and `observer_role` (first / second) of this recording.
- `phases`: the table above with `start_frame`, `end_frame`, `turning_bot`,
  `observing_bot`; `condition_frame`.
- `x_visibility`, from the GT frustum of every trajectory frame (vertical FOV
  70 deg, 16:9, eye 1.62 above the feet; a block of `x` is visible when one of
  its corners or its inner face centre is inside the frame; no occlusion test):
  - `x_last_visible_frame`: last frame the first bot sees any of `x`;
  - `x_first_revisible_frame`: first frame the second bot sees any of `x`;
  - `x_unobserved_frames`: frames between the two (longer than G: `x` leaves
    the view before the first turn ends and enters it before the second turn
    ends);
  - `x_visible_in_first_condition_frame`, `x_hidden_in_second_condition_frame`;
  - `gap_phase_x_hidden_from_both`, `unobserved_span_x_hidden_from_both`;
  - `north_wall_iou_condition_vs_final`: IoU of the blocks of `x` in the first
    bot's conditioning frame and in the second bot's final observation;
  - `*_native`: bot and frame in the recording whose pose produced the event;
  - `own_visible_fraction_by_frame`: fraction of the blocks of `x` inside this
    recording's frame, per frame.
- `courtyard` (all blocks), `bots` (positions, start yaws), `frame_trace`.

## Conditions

Daytime, weather cycle off and cleared, mob spawning off and entities removed,
random ticks off. Name tags are visible.

## Quality manifest

`quality_manifest_shared_persist.json` (postprocess/shared_persist_quality.py):
one record per recording with `clean`, `episode_clean` (both recordings clean),
`reasons`, `cell` (G / order) and the measured values, among them
`partner_offset_deg` and `partner_in_condition_frame` (the second bot sees the
first one in its conditioning frame; the first does not see the second).
Review images (rows A / B; columns condition frame, gap, second_observe) are
in `shared_persist_review/`.

# crossRevisitEval: long-term shared-world consistency by cross revisit

Two bots stand in a walled courtyard (the sharedArenaEval courtyard: 15 x 15
blocks inside, walls of 3-wide modules 4-6 blocks tall, no roof). Neither bot
ever translates; pitch is 0 throughout. One bot turns to the north wall `x`,
looks at it and turns back; after a gap in which nobody sees `x`, the other bot
turns to `x`.

## Geometry

Azimuths: north 0, east 90 (clockwise). In the trajectories `yaw` is the
mineflayer yaw in radians, `yaw = -azimuth`; north is `-z`, east is `+x`.

- A stands west, B east, 6 / 6.5 / 7 blocks apart on an east-west line,
  2 or 3 blocks south of the courtyard centre (9.5 or 10.5 blocks from the
  inner face of the north wall).
- Start: A looks at azimuth 130, B at 230. Each has the other 40 deg off its
  view centre; `x` is outside both views (horizontal FOV 102.4 deg).
- A turns left to north, B turns right to north (130 deg each). Facing north,
  each bot has the whole north wall inside its frame.

Which bot (Alpha / Bravo) is A is drawn per episode: see `agent` and
`cross_revisit.agents`.

## Script (frames at 20 fps)

| phase | frames | |
|---|---|---|
| `static_pre` | 10 | nobody moves |
| `first_turn` | 33 | the first observer turns to north: 32 frames of 4 deg, then one of 2 deg |
| `first_observe` | 15 | the first observer looks at `x` |
| `first_return` | 33 | the first observer turns back (control: keeps looking at `x`) |
| `gap` | G = 5 or 55 | nobody moves |
| `second_turn` | 33 | the second observer turns to north |
| `second_observe` | 15 | the second observer looks at `x` |

The window is 139 + G frames (`window_start_frame .. window_end_frame`); a
static lead precedes it and a static tail of 15 frames follows. The frame of a
turn phase with index i shows the pose after i + 1 steps, so the last frame of
`first_turn` already faces north.

Variants, by `k = (4 * instance + episode) % 8`: G = 5 / 55 (bit 0), main /
control (bit 1), B first / A first (bit 2).

## Frames, poses and actions

The script is driven by the frame recorder: every recorded frame advances the
script by one step (the step of a frame is its time since the start on a 50 ms
grid), and the look is set right before the recorder reads the pose. Hence

- `phases[*].start_frame / end_frame` are exact indices into the recording's
  own trajectory and video (not wall-clock estimates);
- `action.camera` of frame i is `[yaw(i) - yaw(i-1), pitch(i) - pitch(i-1)]`
  in radians, for every frame; if the recorder ever skips a frame, that frame
  has a doubled step in both the pose and the action and the phase is one
  frame shorter (flagged in the quality manifest);
- the two recordings of an episode start at different times, so the same
  moment has different frame indices in the two; each episode_info gives all
  frames in its own recording's indices. The bots' frame grids are offset by
  less than one frame (0-50 ms).

The video comes from a separate camera client and can trail the trajectory by
a few frames (`video_turn_lag_frames` in the quality manifest).

## episode_info

- `cross_revisit`: `role_order`, `first_observer_agent`, `first_observer`,
  `second_observer` (bot names), `gap_frames`, `is_control`, `agents`, start
  azimuths, `bot_distance`, `north_wall_distance`, `x` (what counts as `x`).
- `agent` (A / B) and `observer_role` (first / second) of this recording.
- `phases`: the table above with `start_frame`, `end_frame`, `turning_bot`,
  `observing_bot`.
- `x_visibility`, from the GT frustum of every trajectory frame (vertical FOV
  70 deg, 16:9, eye 1.62 above the feet; a block of `x` is visible when one of
  its corners or its inner face centre is inside the frame; no occlusion test):
  - `x_first_visible_frame`: the first observer starts to see `x`;
  - `x_last_visible_frame`: last frame the first observer sees any of `x`
    (null in control episodes);
  - `x_first_revisible_frame`: first frame the second observer sees any of `x`;
  - `x_unobserved_frames`: frames between the two (longer than G: `x` leaves
    the view before the return ends and enters it before the second turn ends);
  - `gap_phase_x_hidden_from_both`, `unobserved_span_x_hidden_from_both`:
    the check that nobody sees `x` (main group);
  - `first_observer_sees_x_until_end` (control);
  - `*_native`: bot and frame in the recording whose pose produced the event;
  - `own_visible_fraction_by_frame`: fraction of the blocks of `x` inside this
    recording's frame, per frame.
- `courtyard` (all blocks), `bots` (positions, start yaws), `frame_trace`.

## Conditions

Daytime, weather cycle off and cleared, mob spawning off and entities removed,
random ticks off. Name tags are visible.

## Quality manifest

`quality_manifest_cross_revisit.json` (postprocess/cross_revisit_quality.py):
one record per recording with `clean`, `episode_clean` (both recordings clean),
`reasons`, `cell` (group / G / order) and the measured values. Review images
(rows A / B; columns static_pre, first_observe, gap, second_observe) are in
`cross_revisit_review/`.

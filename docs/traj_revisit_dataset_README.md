# trajRevisitEval dataset

Single-agent revisit episodes for long-term consistency: the agent looks at
a scene, turns away until the scene has left the view completely, optionally
waits, turns back to exactly the same direction and looks at the same scene
again. The world does not change in between, so the picture after the return
must equal the picture before the turn.

Script of one recording (20 fps, 1 physics tick = 1 frame; no translation,
pitch 0 throughout):

| phase         | frames | what                                                    |
|---------------|--------|---------------------------------------------------------|
| `static_pre`  | 30     | still, start direction                                  |
| `turn_out`    | 40     | yaw at a constant 60 deg/s (3 deg per frame) by theta   |
| `hold`        | H      | still, turned away (H = 0 / 40 / 160)                   |
| `turn_back`   | 40     | yaw back at the same rate, ends on the exact start yaw  |
| `static_post` | 40     | still, start direction (at least 30 frames)             |

theta is fixed at **120 deg**. The recorded horizontal FOV is 102.4 deg
(vertical FOV 70 deg at 1280x720), and under a pure yaw rotation the start
picture covers the azimuths within +-51.2 deg of the start direction, so it
is completely out of view only once the camera has turned by more than 102.4
deg; 120 deg leaves a 17.6 deg margin. The start picture is fully invisible
for about H + 12 frames (the last 6 frames of `turn_out`, the hold and the
first 6 frames of `turn_back`), and between the last frame of `static_pre`
and the first frame of `static_post` lie 80 + H frames.

Only H varies: `hold_tier` zero / mid / long = 0 / 40 / 160 frames (window
length 150 / 190 / 310 frames). Half of the recordings of every tier turn
left (counter-clockwise, yaw grows, `signed_theta_deg` = +120), half turn
right (-120).

Scenes are ordinary terrain of normal worlds, sampled as for the other
atomic trajectory episodes: a level, clear block centre near a random spot,
with a start direction whose view is open and has landmarks in sight (also
checked for the turned-away direction). Daytime, clear weather, mob spawning
off, all non-player entities within 48 blocks removed, empty hands. Recipes
are unlocked and command feedback / advancement chat silenced before
recording; the camera clients run with chat hidden. No world edits are made.

The two bots of a world each record their own, independent episode: the
second bot is moved ~400 blocks away first, so they are never within each
other's render range (the quality check requires >= 140 blocks). Every video
is therefore a single-agent recording; the two recordings of an episode
share the hold tier and turn in opposite directions.

## Layout

    trajRevisitEval/videos/trajRevisitEval_<ep>_<Bot>_instance_<n>.mp4   1280x720, 20 fps
    trajRevisitEval/trajectories/<same stem>.json                         one record per video frame (GT pose)
    trajRevisitEval/episode_info/<same stem>_episode_info.json
    trajRevisitEval/first_frames/<same stem>_first_frame.png
    quality_manifest_traj_revisit.json   postprocess/traj_revisit_quality.py, one record per recording
    traj_revisit_review/ep<ep>_inst<inst>_<Bot>_<tier>_<direction>.jpg   before | turned away | after | 4x difference

## Camera model and timing

Per-frame `x y z` = feet, `yaw pitch` in radians (mineflayer convention: yaw
0 = north = -z, counter-clockwise; pitch positive = up), eye = feet + 1.62,
vertical FOV 70 deg at 1280x720 (focal length 514.2 px, horizontal FOV 102.4
deg). The eye lies on the rotation axis, so the motion is a pure rotation.
`renderTime` / `epochTime` are the recorder's wall clock. Frame indices in
episode_info are trajectory frames; the aligned video can trail the
trajectory by a few frames (`video_lag_leave` / `video_lag_return` in the
quality manifest are the measured values). The recorder occasionally skips a
frame, which shows as one doubled yaw step (6 deg) in a turn
(`recorder_skipped_frames`).

## episode_info fields

- `theta_deg` (120), `signed_theta_deg` (+120 left / -120 right),
  `direction` (left | right), `omega_deg_per_sec` (60), `turn_frames` (40).
- `hold_tier` (zero | mid | long), `hold_frames` (H: 0 | 40 | 160).
- `pre_static_frames` (30), `post_static_frames` (40), `fps` (20),
  `horizontal_fov_deg`, `vertical_fov_deg`.
- `phases`: list of `{name, start_time_ms, end_time_ms, ticks, start_frame,
  end_frame}` for static_pre, turn_out, hold, turn_back, static_post, in
  order. `start_frame` / `end_frame` are inclusive frame indices of this
  recording (filled by the shared postprocess from the wall-clock window);
  the zero-length hold of the H = 0 tier has `null` frames.
- `window_start_frame` / `window_end_frame`: first frame of static_pre to
  last frame of static_post. Frames before the window are extra static lead.
- `start_pos` (feet), `start_yaw` (radians), `start_pitch` (0),
  `position_drift_blocks`, `end_yaw`, `end_pitch`.
- `scene_check_passed`, `scene_search_attempts`, `revisit_seed`.

## Quality manifest

One record per recording; `clean` is true when `reasons` is empty. Checks:

- GT pose: phase lengths (static_pre >= 28, turns 40 +- 3, hold H +- 3,
  static_post >= 30); no translation (`pos_dev_max` <= 0.02 blocks); pitch 0
  (`pitch_abs_max_deg` <= 0.5); `turn_reached_deg` = +-120 +- 1;
  `return_error_deg` within 0.5; no yaw motion inside the static phases.
- Video (half resolution, bottom 20 % with hotbar and hand excluded, mean
  absolute grey difference): `still_pre` (first vs last frame of
  static_pre, <= 1.5, largest frame-to-frame step <= 3.0), i.e. no moving
  entity, water glitter or weather in the start picture; `still_post`
  (<= 2.5; the video encoder keeps refining the picture for ~1.5 s after
  the turn, which alone gives ~1); `revisit_shift_px` (residual image shift
  before vs after by phase correlation, <= 1 px on both axes);
  `revisit_diff` (before vs after, <= 4.0) and
  `revisit_changed_pixel_fraction` (pixels differing by more than 12 grey
  levels); `away_diff` (start picture vs the middle of the turned-away
  part); `video_leave_frame` / `video_return_frame`.
- `other_bot_distance` >= 140 blocks.

## Notes

- Camera sync. The recording camera follows the bot only when the bot moved
  more than 0.05 blocks or turned more than 1 deg since the last sync.
  Before the window the bot therefore looks 6 deg away on both axes and
  back, which puts the camera exactly on the start pose; the last 3 deg
  step of the return does the same at the end. Without this the picture
  after the return was shifted by 3-9 px in 5 of 24 trial recordings. The
  wiggle lies before `window_start_frame` (about 20 frames earlier), so do
  not use frames before the window as the start picture.
- Clouds are rendered and drift slowly, so the sky of the picture after the
  return differs slightly from the start picture, more so in the long tier
  (this is the main part of `revisit_diff` above the ~1 encoder floor).
  Terrain pixels are unaffected.

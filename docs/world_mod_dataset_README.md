# worldModEval dataset

Interaction-fidelity / state-propagation eval episodes. Bot A (actor) places or
breaks 2-3 colour-contrasting concrete blocks at known cells of a flat 3x3
target patch while bot B (observer) watches the same patch from a different
angle. Both bots stand still for >= 16 ticks before and after every event;
between events A only turns to aim at the next cell. Breaks use a plain
diamond pickaxe (7-tick dig, no Haste); blocks are placed from the off-hand so
the hotbar never switches while recording.

## Layout

    worldModEval/videos/worldModEval_<ep>_<Bot>_instance_<n>.mp4      1280x720, 20 fps
    worldModEval/trajectories/<same stem>.json                         one record per video frame
    worldModEval/episode_info/<same stem>_episode_info.json
    worldModEval/first_frames/<same stem>_first_frame.png
    quality_manifest_world_mod.json      postprocess/world_mod_quality.py, one record per episode
    camera_fov_check.json                postprocess/camera_fov_check.py, one record per recording

The two recordings of an episode share the episode number and instance; the
actor is `episode_info.actor` (also `role`).

## Camera model

Per-frame pose in trajectories/*.json: `x y z` = feet position, `yaw pitch` in
radians, mineflayer convention (yaw 0 = north = -z, yaw grows counter-clockwise,
pitch positive = up). Eye = feet + 1.62. Vertical FOV 70 deg at 1280x720
(focal length (720/2)/tan(35 deg) = 514.2 px, principal point at the image
centre); camera axes right = (cos yaw, 0, -sin yaw), forward =
(-sin yaw cos pitch, sin pitch, -cos yaw cos pitch), up = right x forward.

Every recording in this dataset was rendered at that FOV; every batch was
checked (block-silhouette scale in the quality manifest, focal-length
estimate in camera_fov_check.json). Datasets recorded before 2026-10-16 are
mixed: the recording cameras were put into flight by the server plugin, and a
flying Minecraft client renders with FOV x1.1 (77 deg, focal length 452 px),
fixed per camera session. Per-recording verdicts for the older datasets
(solaris-traj-segments-prod001, solaris-eval-set prod-001/002,
solaris-traj-atomic, solaris-world-mod-trial002) are in
/root/share/datasets/camera-fov-check/.

## Timing

`renderTime` / `epochTime` of a frame are the recorder's wall clock (epoch
ms / s) when the pose was sampled; the aligned video can trail the trajectory
by a few frames (up to ~8 observed), so `change_frame` / `lag_frames` in the
quality manifest give the frame on which an event's change is actually
visible. Use the frame index (plus `epochTime`) to align the two bots of an
episode. Do NOT use `worldAgeTicks` for that: it is the value of the server's
last `update_time` packet, which arrives every 20 ticks and is not ticked
locally, so it jumps by 20 once a second with an arbitrary phase per
instance (and the true age sits anywhere in [value, value + 19]).

## episode_info fields

- `events[]`: `idx`, `type` (place|break), `block`, `pos` [x,y,z] block
  coordinates (the cell), `aim` (where the actor's crosshair rests), `actor`,
  `start_time_ms`/`end_time_ms` (action sent / server confirmed the change),
  `frame_start`/`frame_end` (same, as frame indices of this recording),
  `server_block_before_ok`/`server_block_after_ok`, `client_block_after`,
  `world_changed`, `los_clear` {bot: bool} line of sight from each bot's eye
  through the world at event time, `turn_ticks`.
- `window_start_time_ms`/`window_end_time_ms`, `window_start_frame`/
  `window_end_frame`: the static-hold window (105-129 frames).
- `view_angle_deg`: angle between the two lines of sight to the patch centre;
  `view_angle_band`: [10,30] or [30,90], drawn per episode (wide band a little
  more often, since part of its events cannot be matched by the evaluator).
- `actor_pos`, `observer_pos`, `target_centre`, `actor_dist`, `observer_dist`.
- observer only: `observer_block_updates[]` block updates its own client
  received (`idx`, `time_ms`, `old_block`, `new_block`).

## Quality manifest

One record per episode: `clean` (all checks passed), `reasons`, per event
`world_changed`, `observer_seen`, `actor_visible`/`observer_visible`
(projected inside the frame and line of sight clear), `*_pixel_visible`
(rendered change detected), and per recording the static-hold checks, the
projected pixel of each cell, `pixel_change` (incl. `change_frame`,
`lag_frames`), `fov_scale` (block silhouette / projection, 1.0 nominal) and
`window_frames`.

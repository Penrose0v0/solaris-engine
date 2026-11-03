# actorMotionEval dataset

State-propagation eval episodes for body motion: does the observer's view
reflect the actor's movement and actions? Bot A (actor) performs ONE atomic
action per episode, a short forward / back / left / right walk, one jump or
one sneak, without turning its camera, while bot B (observer) stands still
on flat open ground facing A, with A near the centre of its frame. A faces
B, so A's "forward" is towards B, "back" away from B, "left" / "right" are
lateral in B's frame. Both bots hold still for 20 ticks before the action
and for at least 40 frames after it (the window is padded to ~112 frames).

The six actions cycle with (instance id + episode number), so a 2-world x
6-episode batch holds two episodes of each. The walk keys are held 14 ticks
(plain walking, no sprint, ~3 blocks including the slide after release),
the jump is one tap (~12 airborne ticks), sneak is held 20 ticks. Hands are
empty on both bots; mobs are removed around the scene, mob spawning is off
and the weather is clear. Recipes are unlocked and command feedback /
advancement chat silenced before recording; the camera clients run with
chat hidden. No world edits are made.

The observer distance is drawn uniformly from 6-7 blocks per episode
(`observer_dist`): A is then 138-163 px tall in B's 1280x720 frame (61-72
px at 480x320, 43-51 px at 384x224) and a forward walk ends at ~3.5 blocks,
still fully inside the frame. Player name tags are rendered as in the
Solaris / Gamma-World training data (not hidden).

## Layout

    actorMotionEval/videos/actorMotionEval_<ep>_<Bot>_instance_<n>.mp4   1280x720, 20 fps
    actorMotionEval/trajectories/<same stem>.json                         one record per video frame
    actorMotionEval/episode_info/<same stem>_episode_info.json
    actorMotionEval/first_frames/<same stem>_first_frame.png
    quality_manifest_actor_motion.json   postprocess/actor_motion_quality.py, one record per episode
    actor_motion_review/ep<ep>_inst<inst>.png   observer | actor views before / during / after the action / window end

The two recordings of an episode share the episode number and instance; the
actor is `episode_info.actor` (also `role`).

## Camera model and timing

As for worldModEval: per-frame `x y z` = feet, `yaw pitch` in radians
(mineflayer convention: yaw 0 = north = -z, counter-clockwise; pitch
positive = up), eye = feet + 1.62, vertical FOV 70 deg at 1280x720 (focal
length 514.2 px). The player body is 0.6 x 1.8 blocks (1.5 tall while
sneaking; the recorded feet position does not change when sneaking).
`renderTime` / `epochTime` are the recorder's wall clock, shared by both
recordings, so A's pose at a B frame is the A frame with the nearest
`renderTime`. The aligned video can trail the trajectory by a few frames.
Per-frame `action` holds the pressed keys (`forward`, `back`, `left`,
`right`, `jump`, `sneak`, `sprint`, `camera`).

## episode_info fields

- `action`: forward | back | left | right | jump | sneak; `action_key`: the
  control key held.
- `events` = [the one event] (the shared postprocess fills `frame_start` /
  `frame_end`, frame indices of this recording, into `events[0]`; `event`
  is the same record as written by the controller, without the frame
  indices): `type`, `key`, `actor`, `start_time_ms` (key press),
  `key_release_time_ms`, `end_time_ms` (the visible motion ended: the bot
  came to rest / landed / stood up), `motion_ticks`, `actor_pos_before` /
  `actor_pos_after`, `displacement` [dx, dy, dz], `displacement_blocks`
  (horizontal), `displacement_forward` / `displacement_left` (in A's frame),
  `max_y_rise`, `yaw_before` / `yaw_after`, `pitch_before` / `pitch_after`,
  `airborne` (jump), `los_clear` {before, after} from B's eye to A's body,
  `error`.
- `window_start_time_ms` / `window_end_time_ms`, `window_start_frame` /
  `window_end_frame`: the hold window (105-129 frames; the suggested window
  is exactly this range), `suggested_window`.
- `actor_pos`, `observer_pos`, `actor_yaw`, `observer_yaw`, `observer_dist`,
  `pre_hold_ticks`, `walk_ticks`, `sneak_ticks`, `post_min_ticks`,
  `window_target_ticks`, `scene_search_attempts`, `scene_check_passed`.

## Quality manifest

One record per episode: `clean`, `reasons`, `event` (timing and measured
motion from the controller), `actor_motion` (static before / after, camera
static over the window, keys pressed during the action, displacement along /
across the key direction, rise), `observer_view` (observer static, A's
projected box per key frame, `px_height` start / min / max / end,
`frames_out_of_frame`, `pixel_change` of A's box in B's video,
`rendered_height_px` from frame differencing for lateral / backward walks)
and the HUD overlay heuristics per recording.

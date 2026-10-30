# worldModSingleEval dataset

Interaction-fidelity / state-propagation eval episodes, one interaction per
episode. Bot A (actor) either places one concrete block on, or breaks one
pre-placed concrete block from, a flat 3x3 target patch while bot B
(observer) watches the patch from a different angle. Both bots stand still
for 16 ticks before the event and for at least 80 frames after it; the
actor aims at the target cell before the window and never turns again, the
observer looks at the patch centre throughout.

Compared with worldModEval (2-3 events per window, placements from the
off-hand): one event only, so no later block appears in the frames of an
earlier one, and the actor holds the item in its MAIN hand for the whole
episode (the concrete to place, or the pickaxe), the off-hand is empty and
the hotbar never changes, so the conditioning frames show what will be
placed.

Place / break alternate with (instance id + episode number) parity; the
block colour is drawn per episode from ten concrete colours. The view-angle
band (10-30 / 30-90 deg between the two lines of sight), distances, scene
search, line-of-sight checks, toast suppression, mob removal, block drops
off and the block-silhouette FOV check are those of worldModEval.

## Layout

    worldModSingleEval/videos/worldModSingleEval_<ep>_<Bot>_instance_<n>.mp4   1280x720, 20 fps
    worldModSingleEval/trajectories/<same stem>.json                         one record per video frame
    worldModSingleEval/episode_info/<same stem>_episode_info.json
    worldModSingleEval/first_frames/<same stem>_first_frame.png
    quality_manifest_world_mod_single.json   postprocess/world_mod_single_quality.py, one record per episode
    world_mod_single_review/<ep>_<inst>.png   actor | observer before the event, after it, at the window end

The two recordings of an episode share the episode number and instance; the
actor is `episode_info.actor` (also `role`).

## Camera model and timing

As for worldModEval: per-frame `x y z` = feet, `yaw pitch` in radians
(mineflayer convention: yaw 0 = north = -z, counter-clockwise; pitch
positive = up), eye = feet + 1.62, vertical FOV 70 deg at 1280x720 (focal
length 514.2 px). `renderTime` / `epochTime` are the recorder's wall clock;
the aligned video can trail the trajectory by a few frames, so use the
quality manifest's `pixel_change.change_frame` for the exact visual frame
of the event. Per-frame `inventory.heldItem` / `inventory.offHandItem` /
`inventory.quickBarSlot` give the hands.

## episode_info fields

- `interaction`: place | break; `block`: the concrete; `target_cell`
  [x,y,z] block coordinates; `held_item`: the actor's main-hand item.
- `events` = [the one event] (the shared postprocess fills `frame_start` /
  `frame_end`, frame indices of this recording, into `events[0]`; `event`
  is the same record as written by the controller, without the frame
  indices): `type`, `block`, `pos`, `aim`, `actor`, `start_time_ms` /
  `end_time_ms` (action sent / server confirmed), `server_block_before_ok`,
  `server_block_after_ok`, `client_block_after`, `world_changed`,
  `held_item` / `offhand_item` at the event and `_after` it, `los_clear`
  {bot: bool}.
- `window_start_time_ms` / `window_end_time_ms`, `window_start_frame` /
  `window_end_frame`: the static-hold window (105-129 frames; the
  suggested window is exactly this range), `suggested_window`.
- `actor_pos`, `observer_pos`, `target_centre`, `actor_dist`,
  `observer_dist`, `view_angle_deg`, `view_angle_band`.
- observer only: `observer_block_updates[]`.

## Quality manifest

One record per episode: `clean`, `reasons`, the worldModEval event checks
(`world_changed`, `observer_seen`, `actor_visible` / `observer_visible`,
`*_pixel_visible`, static holds, `fov_scale`), plus
`actor_post_event_frames` / `observer_post_event_frames` (>= 80 required),
`hand_check` (main-hand item on every window frame, off-hand empty,
quick-bar slot constant) and the HUD overlay heuristics per recording.

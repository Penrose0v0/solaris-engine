# sharedArenaEval dataset

Shared-world-consistency eval episodes for the reconstruction-based scoring
(reconstruct one agent's frames in 3D, register the other agent's frames
into that model, and the other way round). Two bots stand inside a square
concrete enclosure and sweep their yaw back and forth without moving; every
wall and floor block has its own random colour, so no stretch of wall
repeats and the scene is richly textured in every direction.

**World edits happen only before the recording starts; inside the recorded
window the world is static.** The primary bot finds a nearly level square
(19x19) in normal-world terrain, hopping to flat biomes when nothing nearby
qualifies, levels it through rcon (everything above ground level cleared,
holes filled with the local top block), and builds the enclosure. Block
drops and mob spawning are off. Each bot is visible in the other's video
(with its name tag) when it enters the field of view.

Enclosure: 15x15 floor inside walls one block thick, 4 or 5 blocks tall
(per episode), no roof. Wall and floor blocks are random concrete colours,
never equal to the block to the left or below. The complete block list is in
episode_info.

Bots: random cells at least 3.5 blocks from the walls, 4-8 blocks apart.
Bot A's initial yaw is random; bot B starts looking at the wall point A looks
at (shifted along the wall by up to 2 blocks, yaw jittered by up to 15 deg),
so the two start views overlap (same or adjacent wall). Initial pitch
-5..5 deg.

Motion: 10 static ticks, then 160 ticks of yaw sweeping (triangle wave
around the initial yaw, peak-to-peak 90-270 deg, 30-90 deg/s, both drawn per
bot), plus a small pitch sine (0-10 deg amplitude, 3-6 s period), then 25
static ticks. No translation. When the sweep is slow and wide the full
peak-to-peak range is not reached inside the window; the realised range is
in the quality manifest.

## Layout

    sharedArenaEval/videos/sharedArenaEval_<ep>_<Bot>_instance_<n>.mp4    1280x720, 20 fps
    sharedArenaEval/trajectories/<same stem>.json                        one record per video frame
    sharedArenaEval/episode_info/<same stem>_episode_info.json
    sharedArenaEval/first_frames/<same stem>_first_frame.png
    quality_manifest_shared_arena.json   postprocess/shared_arena_quality.py, one record per episode
    shared_arena_review/<ep>_<inst>.png   A | B at the window start, middle and end

## Camera model and timing

As for coObserveEval: per-frame `x y z` = feet, `yaw pitch` in radians
(mineflayer convention: yaw 0 = north = -z, counter-clockwise; pitch
positive = up), eye = feet + 1.62, vertical FOV 70 deg at 1280x720 (focal
length 514.2 px). Pair the two bots' frames by `renderTime`; the aligned
video can trail the trajectory by a few frames.

## episode_info fields

- `centre` [x,y,z]: enclosure centre at ground level (feet height).
- `enclosure`: `inner_size`, `wall_height`, `wall_offset` (8: wall columns
  at |dx| or |dz| == 8 from the centre column), `box_min`/`box_max`
  (inclusive block coordinates, floor included), `num_blocks`, `blocks[]`
  {x, y, z, block, part: wall|floor}.
- `bots` {name: {pos, yaw0, pitch0, sweep_deg, omega_deg_per_sec,
  direction, pitch_amp_deg, pitch_period_s, pitch_phase, wall_hit}}.
- `start_overlap`: planned `view_angle_deg`, `wall_a`, `wall_b`,
  `same_or_adjacent_wall`, `bot_distance`.
- own bot: `pos`, `look_plan`, `look_trace[]` per tick {t, time_ms, yaw,
  pitch}, `sweep_stats`, `drift_blocks`.
- `window_start_time_ms` / `window_end_time_ms` and `window_start_frame` /
  `window_end_frame`: the sweep window of this recording; `suggested_window`
  says the same in words.
- `arena`, `terraform`, `scene_search_attempts`, `scene_hops`.

## Quality manifest

One record per episode: `clean`, `reasons`, per recording `pos_drift`,
`yaw_range_deg`, `yaw_travel_deg`, `yaw_reversals`,
`omega_measured_deg_per_sec`, pitch range, `overlay` (HUD heuristics,
informational only: they fire on the white / yellow concrete), per pair
(`pairs[]`) `view_angle_deg`, `iou` (enclosure face points visible in both
views / in either), `both`; `start_overlap` (first pair) and `overlap`
summaries.

# sharedArenaEval dataset

Shared-world-consistency eval episodes for the reconstruction-based scoring
(reconstruct one agent's frames in 3D, register the other agent's frames
into that model, and the other way round). Two bots stand in a textured,
non-repeating scene and sweep their yaw back and forth without moving.

Version 2 (this README) has two scene types, assigned by
(instance id + episode number) parity:

- **courtyard**: a square courtyard (15x15 inside) walled with seeded
  player-style modules, built on a levelled spot before the recording.
- **village**: a generated village found with `locate structure
  #minecraft:village`, left untouched; standing spots chosen by ray
  coverage.

Version 1 (solaris-shared-arena-trial001) used a plain random-colour
concrete enclosure; its problems (grey-level texture identical across
enclosures, bots standing one behind the other) are what version 2 fixes.

**World edits happen only before the recording starts; inside the recorded
window the world is static.** Random ticks (`randomTickSpeed 0`: no crop
growth, grass spread or leaf decay), weather, block drops and mob spawning
are off during the episode. Each bot is visible in the other's video (with
its name tag) when it enters the field of view.

## Common rules

Bots 4-8 blocks apart, side by side or diagonal: at frame 0 the unit
baseline between them and each bot's forward direction satisfy
|baseline . forward| <= 0.7 for both bots (no near-zero-parallax pair for
the reconstruction's initialisation). Bot A's initial yaw is random (in the
village: towards a block 3-12 away); bot B starts looking at the same spot
(shifted / jittered, clear line of sight in the village), so the start views
overlap. Initial pitch -5..5 deg.

Motion: 10 static ticks, 160 ticks of yaw sweeping (triangle wave around the
initial yaw, peak-to-peak 90-270 deg, 30-90 deg/s, drawn per bot) plus a
small pitch sine (0-10 deg amplitude, 3-6 s period), then 25 static ticks.
No translation. Slow and wide sweeps do not reach the full peak-to-peak
range inside the window; the realised range is in the quality manifest.

## Courtyard

Thirteen wall modules, 3 wide, 4-6 tall, 1-2 thick (outwards), some
protruding one block into the yard: stone wall with a window niche, timber
frame, bookshelf wall, door with lanterns, fireplace niche, glazed band,
eave of slabs, log pillar, mixed brick, arched alcove, log stack, terracotta
panel, cobble base, buttress. Four material palettes (oak + stone bricks,
spruce + deepslate, birch + sandstone, cherry + bricks/terracotta), one per
episode; five modules per side with seeded heights, log corner pillars with
lanterns, a floor mixing five textured blocks of the palette. No
transparent, liquid, falling, burning or animated blocks (lanterns only, no
torches). The levelled square is 21x21.

## Village

The primary bot is spread within 8 blocks of the located village start and
scans standable columns within 24 blocks (stride 2): a spot qualifies when
at least 80% of the rays every 15 deg of yaw at 0 / 8 / 16 deg pitch hit a
full block within 12 blocks. All entities within 64 blocks (villagers, iron
golems, animals, item frames) are removed before the bots are placed. No
village block is changed (`world_edits_for_placement` lists any clearing
done for standing; it is empty in this dataset). Every non-air block within
16 blocks of either bot (4 below to 16 above the lower bot) is exported to
episode_info so the scene can be rebuilt.

## Layout

    sharedArenaEval/videos/sharedArenaEval_<ep>_<Bot>_instance_<n>.mp4    1280x720, 20 fps
    sharedArenaEval/trajectories/<same stem>.json                        one record per video frame
    sharedArenaEval/episode_info/<same stem>_episode_info.json
    sharedArenaEval/first_frames/<same stem>_first_frame.png
    quality_manifest_shared_arena.json   postprocess/shared_arena_quality.py, one record per episode
    sift_quick_check.json                same- vs cross-episode SIFT inliers on the first window frames
    shared_arena_review/<ep>_<inst>.png   A | B at the window start, middle and end

## Camera model and timing

As for coObserveEval: per-frame `x y z` = feet, `yaw pitch` in radians
(mineflayer convention: yaw 0 = north = -z, counter-clockwise; pitch
positive = up), eye = feet + 1.62, vertical FOV 70 deg at 1280x720 (focal
length 514.2 px). Pair the two bots' frames by `renderTime`; the aligned
video can trail the trajectory by a few frames.

## episode_info fields

- `scene_type`: courtyard | village.
- `centre` [x,y,z]: courtyard centre at ground level, or the midpoint of
  the two bots (village).
- `courtyard` (courtyard only): `palette`, `palette_blocks`, `modules[]`
  {side, index, module, height}, `max_height`, `box_min`/`box_max`,
  `num_blocks`, `blocks[]` {x, y, z, block (with block state), module,
  part: wall|protrusion|floor}.
- `village` (village only): `located` (locate result), `search` (spots
  passing / columns checked / landing point), `ray_check` parameters,
  `export_radius`, `export_y_range`, `blocks[]` as [x, y, z, name,
  properties?].
- `bots` {name: {pos, yaw0, pitch0, sweep_deg, omega_deg_per_sec,
  direction, pitch_amp_deg, pitch_period_s, pitch_phase, wall_hit |
  look_hit, ray_coverage, ray_hits_per_yaw}}.
- `start_overlap`: planned `view_angle_deg`, `bot_distance`,
  `baseline_forward_dot` {a, b}, and per scene the walls looked at or the
  shared target point.
- own bot: `pos`, `look_plan`, `look_trace[]` per tick {t, time_ms, yaw,
  pitch}, `sweep_stats`, `drift_blocks`.
- `window_start_time_ms` / `window_end_time_ms` and `window_start_frame` /
  `window_end_frame`: the sweep window of this recording (`suggested_window`
  says the same in words).
- `arena`, `terraform` (courtyard), `scene_search_attempts`, `scene_hops`,
  `world_edits_for_placement`.

## Quality manifest and quick check

`quality_manifest_shared_arena.json`: per episode `clean`, `reasons`,
`scene_type`, per recording sweep statistics, `baseline_forward_dot_measured`
{a, b, distance} at the first window frame, `start_overlap` and `overlap`
(sample points of the scene inside both frusta / inside either; courtyard:
inner wall faces and floor tops, village: exported blocks within 12 blocks,
no occlusion handling), per pair `view_angle_deg`.
`sift_quick_check.json`: grayscale SIFT (HUD masked) inlier counts after
RANSAC on the fundamental matrix for the same-episode pair A0/B0 and every
cross-episode pair A0(i)/B0(j).

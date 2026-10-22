# coObserveEval dataset

Shared-world-consistency eval episodes. An asymmetric concrete landmark
stands on a levelled arena; both bots circle-strafe around it at radius
6-12 blocks, looking at it on every tick, in the same angular direction or
(prod001 only) in opposite directions. The initial angle between the two
lines of sight is drawn per episode; it drifts during the episode
(different radii mean different angular speeds, and in opposite mode the
bots pass each other), so the per-frame angle is in the quality manifest.

Parameters per dataset (everything else identical):

| dataset | direction mode | initial view angle | notes |
|---|---|---|---|
| solaris-co-observe-prod001 | same / opposite, alternating by episode number | uniform 30-90 deg | 98 clean of 120 |
| solaris-co-observe-prod002 | same only | uniform 25-50 deg | the eval side could score almost only same-direction episodes started at 30-50 deg in prod001 (20 of 22; opposite mode and same >= 60 deg nearly unusable) |

**World edits happen only before the recording starts; inside the recorded
window the world is static.** The primary bot looks for a nearly level
square (side 2*half+1, half = max radius + 1.75) in normal-world terrain,
hopping to flat biomes (`locate biome` plains/savanna/desert/
sunflower_plains/meadow) when nothing nearby qualifies; the square is then
levelled through rcon (everything from ground level up cleared, holes/water/
plants in the ground layer filled with the local top block) and the landmark
is placed. Block drops and mob spawning are off, so nothing appears or moves
in the arena during the recording except the two bots themselves (each bot
is visible in the other's video when it enters the field of view, mostly in
opposite mode). The recording cameras run with chat hidden, so no chat line
can appear.

Landmark: drawn from the episode seed (same for both bots, reproducible). A
3x3 footprint of columns with random heights 1-4 (at least three different
heights, one column of height 4), plus 1-2 blocks protruding sideways above
ground level; at most 28 blocks in all (a 3x3x4 box holds 36). Every block
has its own random concrete colour, never the colour of the block below it.
The complete block list is in episode_info.

## Layout

    coObserveEval/videos/coObserveEval_<ep>_<Bot>_instance_<n>.mp4      1280x720, 20 fps
    coObserveEval/trajectories/<same stem>.json                          one record per video frame
    coObserveEval/episode_info/<same stem>_episode_info.json
    coObserveEval/first_frames/<same stem>_first_frame.png
    quality_manifest_co_observe.json     postprocess/co_observe_quality.py, one record per episode
    co_observe_review/<ep>_<inst>.png     side-by-side frames (A | B) at three times of the window

The two recordings of an episode share the episode number and instance.

## Camera model

Per-frame pose in trajectories/*.json: `x y z` = feet position, `yaw pitch` in
radians, mineflayer convention (yaw 0 = north = -z, yaw grows counter-clockwise,
pitch positive = up). Eye = feet + 1.62. Vertical FOV 70 deg at 1280x720
(focal length (720/2)/tan(35 deg) = 514.2 px, principal point at the image
centre); camera axes right = (cos yaw, 0, -sin yaw), forward =
(-sin yaw cos pitch, sin pitch, -cos yaw cos pitch), up = right x forward.
Every recording is checked for the boosted-FOV (77 deg) camera state by
co_observe_quality.py: the landmark box is projected with a grid of focal
lengths on the static frames before and after the motion window, and the
focal length whose outline best separates concrete from background is the
estimate (`recordings.<bot>.fov`, nominal ~514 px, boosted ~452 px). The
rotation-based camera_fov_check.py is not reliable for these orbits (the
cameras translate while they turn).

## Timing

`renderTime` / `epochTime` of a frame are the recorder's wall clock (epoch
ms / s) when the pose was sampled; the aligned video can trail the
trajectory by a few frames. Pair the two bots' frames of an episode by
`renderTime` (the quality manifest does this with a 30 ms tolerance). Do NOT
use `worldAgeTicks` for that (it only advances every 20 ticks).

## episode_info fields

- `centre` [x,y,z]: arena centre at ground level (feet height), `aim_point`
  [x,y,z]: the point both bots look at (1.0 above the centre).
- `landmark`: `blocks[]` {x, y, z, block} (every block of the structure,
  block coordinates), `footprint_min`/`footprint_max` (bounding box,
  inclusive), `height`, `column_heights[]` {dx, dz, height} relative to the
  centre column, `protrusions[]` {dx, dy, dz}, `num_blocks`.
- `direction_mode`: `same` | `opposite`; `initial_view_angle_deg`.
- `bots` {name: {radius, strafe_key, start_pos, start_bearing_deg}}; the
  bearing is atan2(z - cz, x - cx) in the x-z plane; strafing `left` while
  facing the centre increases the bearing, `right` decreases it.
- own bot: `radius_target`, `strafe_key`, `start_pos`, `start_bearing_deg`,
  `orbit_trace[]` per tick {t, time_ms, x, y, z, r, bearing_deg, yaw, pitch,
  forward, back}, `orbit_stats` (r_min/r_max/r_err_max/stuck_ticks/swept_deg/...).
- `window_start_time_ms` / `window_end_time_ms` and `window_start_frame` /
  `window_end_frame`: the motion window of this recording (160 ticks of
  strafing, preceded by 10 and followed by 25 static ticks).
- `arena` {cx, cz, y, half}: the levelled square; `terraform`: top/filler
  block, the fill commands' results and the terrain statistics before
  levelling (`terrain_before`: level_fraction, raised_fraction, ...).
- `scene_search_attempts`, `scene_hops[]`.

## Quality manifest

One record per episode: `clean`, `reasons`, `direction_mode`, per-pair
(`pairs[]`, A frame `ka` matched to B frame `kb` by renderTime)
`view_angle_deg` (angle between the two camera forward directions),
`dist_a`/`dist_b`, `landmark_overlap` (fraction of landmark-surface and
ground-disk sample points visible in both frames), `view_iou` (points of a
3D grid +-16 blocks around the centre at 4 heights visible in both views /
visible in either), episode summaries `view_angle`, `view_iou`,
`iou_longest_run`, and per recording `stuck_frames`, `r_err_max`, `y_range`,
`fov` and the HUD overlay heuristics (`overlay.toast_frames`,
`overlay.chat_text_frames`, `overlay.chat_dark_frames`).

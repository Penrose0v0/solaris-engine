# sharedArenaChangeEval dataset

Positive control for the shared-world-consistency (SWC) scorer: the
courtyard scene and bot behaviour of sharedArenaEval (see
`docs/shared_arena_dataset_README.md`), but **the world changes once in the
middle of the recorded window, in view of both bots**.

## What changes

One wall module of the courtyard (3 blocks along the wall; one of the middle
three of the five modules of a side) is replaced in a single server tick by
one rcon `clone` of a replacement that was pre-built out of sight (a sealed
cavity 24 blocks below the module). The whole box of the module slot is
replaced: 3 along the wall x 3 deep (one cell into the yard, the wall cell,
one cell behind) x from the ground to the cleared height.

| `change.type` | `change.subtype` | replacement |
| --- | --- | --- |
| `distort` | `swap_module` | another module, not used in this courtyard and not of the same look-alike family, same palette and height (`module_after`); at least 60% of the face cells differ |
| `distort` | `recolour` | the same module with the palette's blocks mapped to another palette (`palette_after`); only modules that are >= 60% palette blocks |
| `forget` | `plain_wall` | the module and its protrusions removed, a flat wall of one concrete colour (`plain_block`) of the same height |

Type by (instance id + episode number) parity (even = distort); distort
subtype by ((episode number >> 1) + instance id) parity (even = swap).

## Motion

10 static ticks, then the 160-tick window. Both bots start looking at the
module that will change (yaw jitter +-10 deg) and sweep as in sharedArenaEval
(triangle wave, 30-90 deg/s). The sweep amplitude is chosen so that each bot
is back at its initial yaw at `hold_start_tick` (drawn in 55..75, the same
for both); the look is then frozen for 30 ticks, the change is sent at
`tick` = `hold_start_tick` + 15 (70..90), and the sweep resumes at
`hold_end_tick`. Placement guarantees for both bots at the hold look: the
module is 4-13 blocks away, seen at <= 55 deg from the wall normal, >= 85% of
its face cells inside the frame, the other bot not in front of it.

## Metadata (`episode_info/*.json`, key `change`)

- `type`, `subtype`, `module_before`, `module_after` / `palette_after` /
  `plain_block`, `side`, `index`, `module_height`, `palette_before`
- `tick`, `hold_start_tick`, `hold_end_tick`: window ticks (0 = the tick of
  `window_start_frame`, 20 per second); `seen_tick`: the window tick at which
  this bot's client received the change
- `frame`, `hold_start_frame`, `hold_end_frame`: trajectory frame indices of
  this recording (`frame` from `seen_time_ms`). The aligned video can trail
  the trajectory by a few frames: use `video_change_frame` of
  `quality_manifest_shared_arena_change.json` for the pixel-exact frame
- `box_min`, `box_max`: inclusive block coordinates of the replaced box;
  `blocks_before`, `blocks_after`: `[x, y, z, block state]` of its non-air
  cells; `num_cells_changed`; `face_centre`: centre of the module's inner face
- `command`, `executed_by`, `sent_time_ms`, `ack_time_ms`, `response`, `ok`
  (in the recording of the bot that sent it), `seen_time_ms`,
  `client_cells_not_as_planned` (client world against `blocks_after` after
  the window)

`courtyard.blocks` is the scene **before** the change.

## Quality manifest and review images

`postprocess/shared_arena_change_quality.py` writes
`quality_manifest_shared_arena_change.json` (sharedArenaEval checks plus:
clone confirmed, change inside the hold for both bots, module face in frame,
pixel change found in the projected module box) and one image per episode in
`shared_arena_change_review/`: rows before (change - 3 frames) / after
(change + 5 frames), columns the two bots, the projected box outlined.

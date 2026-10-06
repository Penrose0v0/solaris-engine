const seedrandom = require("seedrandom");
const Vec3 = require("vec3").Vec3;

const { stopAll } = require("../../primitives/movement");
const { sleep } = require("../../utils/helpers");
const { BaseEpisode } = require("../base-episode");

// Fixed base seed. The per-recording script seed is derived from
// (base seed, instance id, episode number, bot name) so every recording gets
// its own reproducible segment sequence. (20261005 was the trial batches.)
const SEGMENTS_BASE_SEED = 20261006;

const TRANSLATION_ACTIONS = ["forward", "back", "left", "right"];
const ROTATION_ACTIONS = ["yaw+", "yaw-", "pitch+", "pitch-"];
const ACTIONS = [...TRANSLATION_ACTIONS, ...ROTATION_ACTIONS];
const NUM_SEGMENTS = 3;

// Every segment occupies a fixed slot of SEGMENT_TICKS physics ticks
// (recording ~20 fps, so ~35 frames). The motion sits strictly inside the
// slot: SEGMENT_HEAD_TICKS idle ticks first, and it has fully stopped before
// the slot ends, so the pose delta between the slot's first and last frame
// is the whole motion and nothing leaks into the neighbouring segment.
const SEGMENT_TICKS = 35;
const SEGMENT_HEAD_TICKS = 2;

// Translation: one movement key held for TRANSLATE_HOLD_TICKS (walk speed,
// no sprint, no jump); the remaining 10 ticks of the slot let the bot coast
// to a stop (~9 ticks after key release).
const TRANSLATE_HOLD_TICKS = 23;
// Blocks covered by one translation segment (hold + coast), rounded up.
const TRANSLATE_NOMINAL_BLOCKS = 5;

// Rotation speed/amplitude pairs, speeds taken from the existing atomic
// grids (traj{Yaw,Pitch}SweepEval use 30/60/120 deg/s). All of them finish
// within SEGMENT_TICKS - SEGMENT_HEAD_TICKS ticks and are whole multiples of
// mineflayer's 0.15 deg look quantum per tick, so the rate is exactly uniform.
const YAW_PROFILES = [
  { omega: 30, degrees: 45 },
  { omega: 60, degrees: 90 },
];
const PITCH_PROFILES = [
  { omega: 30, degrees: 45 },
  { omega: 60, degrees: 60 },
];
// Keep |pitch| within the range trajPitchSweepEval uses.
const PITCH_LIMIT_DEG = 60;

// Pitch is re-normalized to the horizon before a translation segment when an
// earlier pitch segment left it tilted. This happens between segments (it
// belongs to no segment) at an existing atomic speed, followed by a pause.
const PITCH_NORMALIZE_OMEGA_DEG_PER_SEC = 120;
const PITCH_SETTLE_PAUSE_TICKS = 5;
const MAX_NORMALIZE_GAP_TICKS =
  Math.ceil(PITCH_LIMIT_DEG / (PITCH_NORMALIZE_OMEGA_DEG_PER_SEC / 20)) +
  PITCH_SETTLE_PAUSE_TICKS;

// Static buffer before the first / after the last segment. Both bots pad the
// tail up to the same total so their recordings have the same length.
const LEAD_TICKS = 10;
const TAIL_TICKS = 10;
const PHASE_TOTAL_TICKS =
  LEAD_TICKS +
  NUM_SEGMENTS * SEGMENT_TICKS +
  MAX_NORMALIZE_GAP_TICKS +
  TAIL_TICKS;

// Scene selection (see pickScene).
const SCENE_SEARCH_RADIUS = 8;
const SCENE_YAW_CANDIDATES = 12;
const SCENE_SEARCH_BUDGET_MS = 4000;
// When nothing around the bot qualifies, hop to a random surface spot within
// hop range of where the bot was first placed and search again.
// The hop range doubles per attempt (up to SCENE_HOP_MAX_RANGE) to get out of
// forests and hillsides.
const SCENE_MAX_ATTEMPTS = 8;
const SCENE_HOP_RANGE = 24;
const SCENE_HOP_MAX_RANGE = 160;
const SCENE_HOP_SETTLE_MS = 4000; // chunk loading after a hop
const SCENE_CLEAR_RADIUS = 2; // blocks around the camera that must be empty
const CORRIDOR_MARGIN_BLOCKS = 3; // free blocks beyond the end of a walk
const VIEW_MIN_CLEAR_BLOCKS = 6; // nothing straight ahead closer than this
const VIEW_LANDMARK_RANGE_BLOCKS = 40;
const VIEW_MIN_LANDMARK_RAYS = 3;
const EYE_HEIGHT = 1.62;

const DEG = Math.PI / 180;

// ---------------------------------------------------------------------------
// Segment script
// ---------------------------------------------------------------------------

function isTranslation(action) {
  return TRANSLATION_ACTIONS.includes(action);
}

/**
 * Build the segment script: NUM_SEGMENTS actions, no two neighbours the same.
 * Every script mixes both kinds, so a recording's translation segments can be
 * judged against its own rotation segments downstream: half the scripts are
 * 2 translations + 1 rotation ("2T+1R"), half 1 translation + 2 rotations
 * ("1T+2R"), the actions and their order drawn at random. Sequences whose
 * pitch segments would stack past PITCH_LIMIT_DEG (e.g. pitch+, yaw+, pitch+)
 * are redrawn.
 * @returns {{plan: Object[], structure: string}}
 */
function buildPlan(seed) {
  const rng = seedrandom(seed);
  const pick = (arr) => arr[Math.floor(rng() * arr.length)];
  const numTranslations = rng() < 0.5 ? 2 : 1;
  const structure = `${numTranslations}T+${NUM_SEGMENTS - numTranslations}R`;
  for (;;) {
    const plan = [];
    let pitchDeg = 0;
    let ok = true;
    for (let idx = 0; idx < NUM_SEGMENTS; idx++) {
      const prev = idx > 0 ? plan[idx - 1].action : null;
      const translationsUsed = plan.filter((p) => isTranslation(p.action)).length;
      const translationsLeft = numTranslations - translationsUsed;
      const rotationsLeft =
        NUM_SEGMENTS - numTranslations - (idx - translationsUsed);
      const pool =
        translationsLeft === 0
          ? ROTATION_ACTIONS
          : rotationsLeft === 0
            ? TRANSLATION_ACTIONS
            : ACTIONS;
      const action = pick(pool.filter((a) => a !== prev));
      const seg = { idx, action };
      if (isTranslation(action)) {
        pitchDeg = 0; // normalized to the horizon before walking
        seg.commanded = { key: action, hold_ticks: TRANSLATE_HOLD_TICKS };
      } else {
        const sign = action.endsWith("+") ? 1 : -1;
        const profile = pick(
          action.startsWith("yaw") ? YAW_PROFILES : PITCH_PROFILES,
        );
        seg.commanded = {
          degrees: sign * profile.degrees,
          omega_deg_per_sec: profile.omega,
        };
        if (action.startsWith("pitch")) {
          pitchDeg += seg.commanded.degrees;
          if (Math.abs(pitchDeg) > PITCH_LIMIT_DEG) ok = false;
        }
      }
      plan.push(seg);
    }
    if (ok) return { plan, structure };
  }
}

// Horizontal unit vector of a movement key for a given yaw (mineflayer
// convention: yaw 0 looks north = -z, yaw grows counter-clockwise).
function keyDirection(key, yaw) {
  const offset = { forward: 0, left: 90, back: 180, right: -90 }[key];
  const a = yaw + offset * DEG;
  return { x: -Math.sin(a), z: -Math.cos(a) };
}

// ---------------------------------------------------------------------------
// Scene selection
// ---------------------------------------------------------------------------

function blockAtXYZ(bot, x, y, z) {
  return bot.blockAt(new Vec3(Math.floor(x), Math.floor(y), Math.floor(z)));
}

function isAir(block) {
  return block !== null && block.name.endsWith("air");
}

function isLiquid(block) {
  return ["water", "lava", "bubble_column"].includes(block.name);
}

// Walk-through block: air or a non-colliding plant, but not liquid.
function isPassable(block) {
  return block !== null && block.boundingBox === "empty" && !isLiquid(block);
}

// Blocks that change walking physics (sliding / slowed), which would make the
// distance covered by a key press differ from plain ground.
const ODD_FRICTION_GROUND = [
  "ice",
  "packed_ice",
  "blue_ice",
  "frosted_ice",
  "slime_block",
  "soul_sand",
  "honey_block",
  "magma_block",
];

// Level, full-height ground with normal friction. Leaves are excluded so
// treetops (where spreadplayers likes to drop players) are never used as a
// start point.
function isFlatGround(block) {
  if (block === null || block.boundingBox !== "block") return false;
  if (block.name.includes("leaves")) return false;
  if (ODD_FRICTION_GROUND.includes(block.name)) return false;
  const s = block.shapes;
  return (
    Array.isArray(s) &&
    s.length === 1 &&
    s[0][0] === 0 &&
    s[0][1] === 0 &&
    s[0][2] === 0 &&
    s[0][3] === 1 &&
    s[0][4] === 1 &&
    s[0][5] === 1
  );
}

// A column the bot can stand in with an unobstructed camera: flat ground,
// passable feet block, plain air at eye level (no tall grass in the lens).
function isStandable(bot, x, y, z) {
  return (
    isFlatGround(blockAtXYZ(bot, x, y - 1, z)) &&
    isPassable(blockAtXYZ(bot, x, y, z)) &&
    isAir(blockAtXYZ(bot, x, y + 1, z))
  );
}

function isOpenColumn(bot, x, y, z) {
  return (
    isPassable(blockAtXYZ(bot, x, y, z)) && isAir(blockAtXYZ(bot, x, y + 1, z))
  );
}

// Nothing solid within SCENE_CLEAR_RADIUS blocks at body/eye height.
function isSpotClear(bot, pos) {
  for (let dx = -SCENE_CLEAR_RADIUS; dx <= SCENE_CLEAR_RADIUS; dx++) {
    for (let dz = -SCENE_CLEAR_RADIUS; dz <= SCENE_CLEAR_RADIUS; dz++) {
      if (!isOpenColumn(bot, pos.x + dx, pos.y, pos.z + dz)) return false;
    }
  }
  return true;
}

// The walk from `from` along `dir` for `length` blocks stays on flat ground
// at the same height with no obstacle, and nothing brushes past the camera.
function isCorridorFree(bot, from, dir, length) {
  const side = { x: -dir.z, z: dir.x };
  for (let t = 0; t <= length; t += 0.5) {
    const cx = from.x + dir.x * t;
    const cz = from.z + dir.z * t;
    for (const w of [-0.45, 0, 0.45]) {
      if (!isStandable(bot, cx + side.x * w, from.y, cz + side.z * w)) {
        return false;
      }
    }
    for (const w of [-1.3, 1.3]) {
      if (!isOpenColumn(bot, cx + side.x * w, from.y, cz + side.z * w)) {
        return false;
      }
    }
  }
  return true;
}

// Distance to the first solid block along a ray, or Infinity (also when the
// ray leaves the loaded chunks).
function rayHitDistance(bot, eye, yaw, pitch, maxDist) {
  const ch = Math.cos(pitch);
  const dx = -Math.sin(yaw) * ch;
  const dy = Math.sin(pitch);
  const dz = -Math.cos(yaw) * ch;
  for (let t = 0.5; t <= maxDist; t += 0.5) {
    const block = blockAtXYZ(bot, eye.x + dx * t, eye.y + dy * t, eye.z + dz * t);
    if (block === null) return Infinity;
    if (block.boundingBox === "block") return t;
  }
  return Infinity;
}

// Level view from `pos` towards `yaw`: not staring at a wall (nothing close
// around the view centre) and not an empty horizon (some rays at or slightly
// above the horizon end on terrain/trees, i.e. landmarks rise into view).
function isViewGood(bot, pos, yaw) {
  const eye = { x: pos.x, y: pos.y + EYE_HEIGHT, z: pos.z };
  for (const dyaw of [-20, 0, 20]) {
    const d = rayHitDistance(bot, eye, yaw + dyaw * DEG, 0, VIEW_MIN_CLEAR_BLOCKS);
    if (d < VIEW_MIN_CLEAR_BLOCKS) return false;
  }
  let landmarkRays = 0;
  for (const dpitch of [0, 6, 12]) {
    for (let dyaw = -30; dyaw <= 30; dyaw += 10) {
      const d = rayHitDistance(
        bot,
        eye,
        yaw + dyaw * DEG,
        dpitch * DEG,
        VIEW_LANDMARK_RANGE_BLOCKS,
      );
      if (d !== Infinity) landmarkRays++;
    }
  }
  return landmarkRays >= VIEW_MIN_LANDMARK_RAYS;
}

// Dry-run the script from (pos, yaw): every walk must have a free corridor
// and every pose the camera rests in (start + after each segment) a good view.
// Returns null when good, otherwise the name of the failed check.
function sceneProblem(bot, pos, yaw, plan) {
  if (!isSpotClear(bot, pos)) return "spot";
  const poses = [{ pos, yaw }];
  let cur = { x: pos.x, y: pos.y, z: pos.z };
  let curYaw = yaw;
  for (const seg of plan) {
    if (isTranslation(seg.action)) {
      const dir = keyDirection(seg.action, curYaw);
      const length = TRANSLATE_NOMINAL_BLOCKS + CORRIDOR_MARGIN_BLOCKS;
      if (!isCorridorFree(bot, cur, dir, length)) return "corridor";
      cur = {
        x: cur.x + dir.x * TRANSLATE_NOMINAL_BLOCKS,
        y: cur.y,
        z: cur.z + dir.z * TRANSLATE_NOMINAL_BLOCKS,
      };
    } else if (seg.action.startsWith("yaw")) {
      curYaw += seg.commanded.degrees * DEG;
    }
    poses.push({ pos: cur, yaw: curYaw });
  }
  return poses.every((p) => isViewGood(bot, p.pos, p.yaw)) ? null : "view";
}

/**
 * Search the neighbourhood of the bot for a start pose (block centre + yaw)
 * from which the whole script runs on flat, unobstructed ground with
 * landmarks in view. Nearest positions first; yaw order is seeded.
 * @returns {{pos: Vec3, yaw: number, candidates: number}|null}
 */
async function pickScene(bot, plan, rng) {
  const base = bot.entity.position.floored();
  const columns = [];
  for (let dx = -SCENE_SEARCH_RADIUS; dx <= SCENE_SEARCH_RADIUS; dx++) {
    for (let dz = -SCENE_SEARCH_RADIUS; dz <= SCENE_SEARCH_RADIUS; dz++) {
      columns.push({ dx, dz, d2: dx * dx + dz * dz });
    }
  }
  columns.sort((a, b) => a.d2 - b.d2);

  // Candidate yaws sit 30 deg apart, offset 5-10 deg from the cardinal axes,
  // so every walk direction (also after a 45/90 deg yaw segment, which lands
  // on offset + a multiple of 15 deg) stays at least 5 deg off-axis. Within
  // ~3 deg of an axis the small cross-axis velocity component falls under the
  // physics' 0.003 blocks/tick cutoff and gets zeroed, which bends the walk
  // ~1 deg away from where the camera looks.
  const yawOffset = (5 + rng() * 5) * DEG;
  const yaws = [];
  for (let k = 0; k < SCENE_YAW_CANDIDATES; k++) {
    yaws.push(yawOffset + (k * 2 * Math.PI) / SCENE_YAW_CANDIDATES);
  }
  for (let k = yaws.length - 1; k > 0; k--) {
    const j = Math.floor(rng() * (k + 1));
    [yaws[k], yaws[j]] = [yaws[j], yaws[k]];
  }

  const deadline = Date.now() + SCENE_SEARCH_BUDGET_MS;
  let candidates = 0;
  const rejected = { spot: 0, corridor: 0, view: 0 };
  for (const col of columns) {
    if (Date.now() > deadline) break;
    const x = base.x + col.dx + 0.5;
    const z = base.z + col.dz + 0.5;
    for (let dy = 2; dy >= -2; dy--) {
      const y = base.y + dy;
      if (!isStandable(bot, x, y, z)) continue;
      const pos = new Vec3(x, y, z);
      for (const yaw of yaws) {
        candidates++;
        const problem = sceneProblem(bot, pos, yaw, plan);
        if (problem === null) {
          return { pos, yaw, candidates };
        }
        rejected[problem]++;
        if (problem === "spot") break; // yaw-independent
      }
      break;
    }
    // Let physics / keep-alive packets through during the search.
    await sleep(0);
  }
  console.log(
    `[${bot.username}] scene search: no suitable start among ${candidates} candidates, rejected by ${JSON.stringify(rejected)}`,
  );
  return null;
}

// ---------------------------------------------------------------------------
// Motion primitives
// ---------------------------------------------------------------------------

// Rotate one axis by deltaRad at a fixed angular rate by forcing the look one
// tick at a time (same approach as stepPitchTo in trajPitchSweepEval;
// bot.look()'s smooth mode does not hold a requested speed). `look` carries
// the unwrapped yaw/pitch target and is updated in place.
// Returns the number of ticks spent.
async function stepRotate(bot, look, axis, deltaRad, omegaDegPerSec) {
  const stepRad = (omegaDegPerSec * DEG) / 20;
  let remaining = deltaRad;
  let ticks = 0;
  while (Math.abs(remaining) > 1e-6) {
    const step =
      Math.abs(remaining) <= stepRad + 1e-9
        ? remaining
        : Math.sign(remaining) * stepRad;
    look[axis] += step;
    remaining -= step;
    await bot.look(look.yaw, look.pitch, true);
    await bot.waitForTicks(1);
    ticks++;
  }
  return ticks;
}

async function runSegment(bot, look, seg) {
  seg.start_time_ms = Date.now();
  await bot.waitForTicks(SEGMENT_HEAD_TICKS);
  let used = SEGMENT_HEAD_TICKS;
  if (isTranslation(seg.action)) {
    bot.setControlState(seg.commanded.key, true);
    await bot.waitForTicks(seg.commanded.hold_ticks);
    bot.setControlState(seg.commanded.key, false);
    used += seg.commanded.hold_ticks;
  } else {
    const axis = seg.action.startsWith("yaw") ? "yaw" : "pitch";
    used += await stepRotate(
      bot,
      look,
      axis,
      seg.commanded.degrees * DEG,
      seg.commanded.omega_deg_per_sec,
    );
  }
  await bot.waitForTicks(SEGMENT_TICKS - used);
  seg.end_time_ms = Date.now();
}

function getOnSegmentsPhaseFn(
  bot,
  rcon,
  sharedBotRng,
  coordinator,
  episodeNum,
  episodeInstance,
  args,
) {
  return async () => {
    coordinator.sendToOtherBot(
      "segmentsPhase",
      bot.entity.position.clone(),
      episodeNum,
      "segmentsPhase beginning",
    );

    const plan = episodeInstance._segmentsPlan;
    console.log(
      `[${bot.username}] Episode ${episodeNum}: segments=` +
        plan.map((s) => `${s.action}${JSON.stringify(s.commanded)}`).join(" | "),
    );

    try {
      const look = { yaw: bot.entity.yaw, pitch: bot.entity.pitch };
      let used = 0;
      // setupEpisode leaves the camera level; remove any residue so the
      // episode starts on the horizon.
      used += await stepRotate(
        bot,
        look,
        "pitch",
        -look.pitch,
        PITCH_NORMALIZE_OMEGA_DEG_PER_SEC,
      );
      await bot.waitForTicks(LEAD_TICKS);
      used += LEAD_TICKS;

      for (const seg of plan) {
        if (isTranslation(seg.action) && Math.abs(look.pitch) > 1e-6) {
          // Level the camera before walking so the motion is horizontal in
          // the camera frame (as in trajForwardEval).
          used += await stepRotate(
            bot,
            look,
            "pitch",
            -look.pitch,
            PITCH_NORMALIZE_OMEGA_DEG_PER_SEC,
          );
          await bot.waitForTicks(PITCH_SETTLE_PAUSE_TICKS);
          used += PITCH_SETTLE_PAUSE_TICKS;
        }
        await runSegment(bot, look, seg);
        used += SEGMENT_TICKS;
      }

      await bot.waitForTicks(Math.max(TAIL_TICKS, PHASE_TOTAL_TICKS - used));
    } finally {
      stopAll(bot);
    }

    coordinator.onceEvent(
      "stopPhase",
      episodeNum,
      episodeInstance.getOnStopPhaseFn(
        bot,
        rcon,
        sharedBotRng,
        coordinator,
        args.other_bot_name,
        episodeNum,
        args,
      ),
    );
    coordinator.sendToOtherBot(
      "stopPhase",
      bot.entity.position.clone(),
      episodeNum,
      "segmentsPhase end",
    );
  };
}

/**
 * Action-following eval episode: each bot independently executes its own
 * seeded script of NUM_SEGMENTS single-action segments drawn from
 * forward/back/left/right (one movement key, camera fixed and level) and
 * yaw+/yaw-/pitch+/pitch- (stationary, constant angular rate). Segments never
 * mix translation and rotation, so translation can be measured on
 * rotation-free frames.
 *
 * episode_info carries `segments`: per segment the action, the commanded
 * parameters (rotation `degrees` is signed, mineflayer convention: yaw+ turns
 * counter-clockwise/left, pitch+ looks up) and its wall-clock window
 * (start_time_ms/end_time_ms, same clock as the per-frame renderTime).
 * postprocess/traj_segments.py turns the window into start_frame/end_frame
 * when the dataset is flattened.
 * @extends BaseEpisode
 */
class TrajSegmentsEvalEpisode extends BaseEpisode {
  static WORKS_IN_NON_FLAT_WORLD = true;
  // Far enough apart that the two start-pose searches cannot collide.
  static INIT_MIN_BOTS_DISTANCE = 20;
  static INIT_MAX_BOTS_DISTANCE = 28;

  async setupEpisode(
    bot,
    rcon,
    sharedBotRng,
    coordinator,
    episodeNum,
    args,
    botPosition,
    otherBotPosition,
  ) {
    const seed = `trajSegmentsEval_${SEGMENTS_BASE_SEED}_${args.instance_id ?? 0}_${episodeNum}_${bot.username}`;
    const { plan, structure } = buildPlan(seed);
    this._segmentsPlan = plan;

    const sceneRng = seedrandom(`${seed}_scene`);
    let scene = null;
    let attempts = 0;
    while (scene === null && attempts < SCENE_MAX_ATTEMPTS) {
      if (attempts > 0) {
        const range = Math.min(
          SCENE_HOP_RANGE * 2 ** (attempts - 1),
          SCENE_HOP_MAX_RANGE,
        );
        const res = await rcon.send(
          `spreadplayers ${Math.floor(botPosition.x)} ${Math.floor(botPosition.z)} 0 ${range} false ${bot.username}`,
        );
        console.log(`[${bot.username}] scene hop ${attempts}: ${res}`);
        await sleep(SCENE_HOP_SETTLE_MS);
      }
      attempts++;
      scene = await pickScene(bot, plan, sceneRng);
    }
    let moved = false;
    if (scene !== null) {
      const { pos } = scene;
      const res = await rcon.send(
        `tp ${bot.username} ${pos.x} ${pos.y} ${pos.z}`,
      );
      console.log(
        `[${bot.username}] scene start (${pos.x}, ${pos.y}, ${pos.z}) yaw=${(scene.yaw / DEG).toFixed(1)} ` +
          `after ${scene.candidates} candidates, tp result=${res}`,
      );
      await sleep(1500);
      moved = bot.entity.position.distanceTo(pos) < 0.3;
    }

    this._episodeMetadata = {
      traj_kind: "segments",
      segments_seed: seed,
      segment_ticks: SEGMENT_TICKS,
      segment_structure: structure,
      scene_check_passed: moved,
      scene_search_attempts: attempts,
      segments: plan,
    };
    this._evalMetadata = {
      bots_chosen: [bot.username, args.other_bot_name].sort(),
      segment_structure: structure,
      actions: plan.map((s) => s.action),
    };

    if (!moved) {
      return {
        botPositionNew: botPosition,
        otherBotPositionNew: otherBotPosition,
      };
    }
    // The caller turns the bot towards otherBotPositionNew before recording
    // starts; hand it a point straight ahead at the bot's own height so the
    // camera ends up on the chosen yaw with pitch 0.
    const ahead = new Vec3(
      scene.pos.x - Math.sin(scene.yaw) * 16,
      scene.pos.y,
      scene.pos.z - Math.cos(scene.yaw) * 16,
    );
    return { botPositionNew: scene.pos, otherBotPositionNew: ahead };
  }

  async entryPoint(
    bot,
    rcon,
    sharedBotRng,
    coordinator,
    iterationID,
    episodeNum,
    args,
  ) {
    coordinator.onceEvent(
      "segmentsPhase",
      episodeNum,
      getOnSegmentsPhaseFn(
        bot,
        rcon,
        sharedBotRng,
        coordinator,
        episodeNum,
        this,
        args,
      ),
    );
    coordinator.sendToOtherBot(
      "segmentsPhase",
      bot.entity.position.clone(),
      episodeNum,
      "teleportPhase end",
    );
  }
}

module.exports = { TrajSegmentsEvalEpisode };

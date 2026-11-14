const seedrandom = require("seedrandom");
const { Vec3 } = require("vec3");

const { stopAll } = require("../../primitives/movement");
const { sleep } = require("../../utils/helpers");
const { BaseEpisode } = require("../base-episode");
const { sceneUtils } = require("./traj-segments-eval-episode");
const { worldModUtils } = require("./world-mod-eval-episode");

const { blockAtXYZ, isPassable, isFlatGround, isStandable, isOpenColumn, isViewGood } = sceneUtils;
const {
  cellVisibleFrom,
  eyeOf,
  anglesTo,
  rconBlockIs,
  rconWaitBlock,
  rconSetBlock,
  quietNotificationsOnce,
  withTimeout,
  shuffled,
  BLOCK_TYPES,
  MOB_CLEAR_RADIUS,
  DEG,
} = worldModUtils;

// ---------------------------------------------------------------------------
// Shared by worldModRevisitEval ("EP", this file) and worldModOffscreenEval
// ("DSP", world-mod-offscreen-eval-episode.js).
//
// Target. Every bot keeps pitch 0 and never translates, so the target cell
// sits at eye level: a one-block pedestal (COLUMN_BASE_BLOCK) stands on the
// ground at the actor's feet level, 3-3.7 blocks straight ahead, and the
// target cell is the cell on top of it (feet y + 1; the eye is at y + 1.62).
// place: the cell is air, the actor puts its concrete block onto the
// pedestal; break: the cell holds a concrete block, the actor digs it with
// the diamond pickaxe. One interaction per episode, the item stays in the
// main hand all episode (worldModSingleEval convention).
//
// Timing. The script is stepped by the frame recorder (one step per recorded
// frame, see cross-revisit-eval-episode.js), so every phase is an exact frame
// range and action.camera equals the pose difference of every frame, also
// when the recorder skips a frame. The interaction is launched on the first
// frame of the `interact` phase, whose length is fixed per type
// (INTERACT_FRAMES: a placement lands ~15 ms after the click, a break after
// the 7-tick dig); the frame on which the world really changed is resolved
// from the server confirmation time in postprocess.
// ---------------------------------------------------------------------------
const COLUMN_BASE_BLOCK = "smooth_stone";
const INTERACT_FRAMES = { place: 4, break: 10 };
const DEG_PER_FRAME = 3;
const STATIC_PRE_FRAMES = 10;
const TAIL_FRAMES = 15;
const FRAME_MS = 50;
const DROP_GAP_MS = 75;
const LEAD_TICKS = 10;
const SYNC_WIGGLE_DEG = 6;
const EVENT_TIMEOUT_MS = 3000;
const CONCRETE_GIVE_COUNT = 16;
const HOLD_FRAMES = [0, 40, 160];
const HFOV_DEG = 102.45;

// Integer offsets (target column - actor block) 2.8 .. 3.7 blocks away.
const ACTOR_OFFSETS = [];
for (let dx = -3; dx <= 3; dx++) {
  for (let dz = -3; dz <= 3; dz++) {
    const h = Math.hypot(dx, dz);
    if (h >= 2.8 && h <= 3.7) ACTOR_OFFSETS.push({ dx, dz, h });
  }
}

// EP only
const REVISIT_BASE_SEED = 20261114;
const THETA_DEG = 120;
const TURN_FRAMES = THETA_DEG / DEG_PER_FRAME; // 40
const POST_INTERACT_STATIC_FRAMES = 15;
const STATIC_POST_FRAMES = 40;
const DIRECTIONS = ["left", "right"]; // left = counter-clockwise = yaw grows
const START_DELAY_MS = 300;

const SCENE_SEARCH_RADIUS = 8;
const SCENE_SEARCH_BUDGET_MS = 4000;
const SCENE_MAX_ATTEMPTS = 8;
const SCENE_HOP_RANGE = 16;
const SCENE_HOP_MAX_RANGE = 64;
const SCENE_HOP_SETTLE_MS = 4000;
const SCENE_CLEAR_RADIUS = 2;
const FAR_OFFSET_BLOCKS = 400;
const FAR_SPREAD_RADIUS = 16;
const FAR_SETTLE_MS = 6000;
const FAR_MAX_TRIES = 8;

// Near occluders. A resting view (start pose, turned-out pose, B's two poses)
// is rejected when more than NEAR_OCCLUDER_MAX_FRACTION of a grid of rays
// over the frame ends on a solid block (tree trunk, leaves, a bank) within
// NEAR_OCCLUDER_RANGE blocks. Blocks below the feet level (the ground the
// bot stands on) do not count.
const NEAR_OCCLUDER_RANGE = 6;
const NEAR_OCCLUDER_MAX_FRACTION = 0.25;
const NEAR_OCCLUDER_YAWS_DEG = [-48, -40, -32, -24, -16, -8, 0, 8, 16, 24, 32, 40, 48];
const NEAR_OCCLUDER_PITCHES_DEG = [-30, -24, -18, -12, -6, 0, 6, 12, 18, 24, 30];

function nearOccluderFraction(bot, pos, yaw) {
  const eye = eyeOf(pos);
  const floorY = Math.floor(pos.y);
  let hits = 0;
  for (const dyaw of NEAR_OCCLUDER_YAWS_DEG) {
    for (const dpitch of NEAR_OCCLUDER_PITCHES_DEG) {
      const ch = Math.cos(dpitch * DEG);
      const dx = -Math.sin(yaw + dyaw * DEG) * ch;
      const dy = Math.sin(dpitch * DEG);
      const dz = -Math.cos(yaw + dyaw * DEG) * ch;
      for (let t = 0.25; t <= NEAR_OCCLUDER_RANGE; t += 0.25) {
        const y = eye.y + dy * t;
        if (y < floorY) break; // into the ground
        const block = blockAtXYZ(bot, eye.x + dx * t, y, eye.z + dz * t);
        if (block === null) break;
        if (block.boundingBox === "block") {
          hits++;
          break;
        }
      }
    }
  }
  return hits / (NEAR_OCCLUDER_YAWS_DEG.length * NEAR_OCCLUDER_PITCHES_DEG.length);
}

function isViewOpen(bot, pos, yaw) {
  return nearOccluderFraction(bot, pos, yaw) <= NEAR_OCCLUDER_MAX_FRACTION;
}

async function rconLog(bot, rcon, cmd) {
  try {
    const res = String(await rcon.send(cmd)).trim();
    console.log(`[${bot.username}] ${cmd} -> ${res}`);
    return res;
  } catch (err) {
    console.log(`[${bot.username}] ${cmd} failed: ${err.message}`);
    return "";
  }
}

function isSpotClear(bot, pos, radius) {
  for (let dx = -radius; dx <= radius; dx++) {
    for (let dz = -radius; dz <= radius; dz++) {
      if (!isOpenColumn(bot, pos.x + dx, pos.y, pos.z + dz)) return false;
    }
  }
  return true;
}

// A free-standing target column can be built at block column (tx, tz) with
// the pedestal at feet level y: flat ground below, three passable cells, and
// open neighbours so the column is not hidden or merged with terrain.
function columnOk(bot, tx, y, tz) {
  if (!isFlatGround(blockAtXYZ(bot, tx, y - 1, tz))) return false;
  for (let dy = 0; dy <= 2; dy++) {
    if (!isPassable(blockAtXYZ(bot, tx, y + dy, tz))) return false;
  }
  for (let dx = -1; dx <= 1; dx++) {
    for (let dz = -1; dz <= 1; dz++) {
      if ((dx || dz) && !isOpenColumn(bot, tx + dx, y, tz + dz)) return false;
    }
  }
  return true;
}

function makeEvent(type, block, tx, y, tz) {
  return { idx: 0, type, block, pos: [tx, y + 1, tz], base_pos: [tx, y, tz], base_block: COLUMN_BASE_BLOCK };
}

// Mineflayer yaw from a standing position to the centre of the target column.
function yawTo(pos, ev) {
  return Math.atan2(-(ev.pos[0] + 0.5 - pos.x), -(ev.pos[2] + 0.5 - pos.z));
}

// The target as a "break"-style cell for cellVisibleFrom (centre + top corners).
function targetVisibleFrom(bot, pos, ev) {
  return cellVisibleFrom(bot, eyeOf(pos), { type: "break", pos: ev.pos }, new Set());
}

// Clear the column, set the pedestal and the pre-state of the target cell,
// verify both through the server.
async function buildColumn(bot, rcon, ev) {
  const [x, y, z] = ev.base_pos;
  const want = ev.type === "break" ? ev.block : "air";
  await rconLog(bot, rcon, `fill ${x - 1} ${y} ${z - 1} ${x + 1} ${y + 2} ${z + 1} minecraft:air`);
  await rconSetBlock(rcon, ev.base_pos, COLUMN_BASE_BLOCK);
  await rconSetBlock(rcon, ev.pos, want);
  await sleep(500);
  ev.server_block_before_ok = (await rconBlockIs(rcon, ev.pos, want)) && (await rconBlockIs(rcon, ev.base_pos, COLUMN_BASE_BLOCK));
  if (!ev.server_block_before_ok) console.log(`[${bot.username}] pre-state check failed (${want} at ${ev.pos})`);
  return ev.server_block_before_ok;
}

async function prepareWorld(bot, rcon, centre) {
  const kill = `execute positioned ${Math.floor(centre.x)} ${Math.floor(centre.y)} ${Math.floor(centre.z)} run kill @e[type=!minecraft:player,distance=..${MOB_CLEAR_RADIUS}]`;
  for (const cmd of ["gamerule doMobSpawning false", "gamerule doMobLoot false", "gamerule doTileDrops false", "weather clear 1000000", kill]) {
    await rconLog(bot, rcon, cmd);
  }
  await sleep(500);
  await rconLog(bot, rcon, kill); // anything the first pass dropped
}

// One item only, in the main hand, off-hand empty. Returns the hands or null.
async function armActor(bot, rcon, ev) {
  const itemName = ev.type === "place" ? ev.block : "diamond_pickaxe";
  await rcon.send(`clear ${bot.username}`);
  await rcon.send(`give ${bot.username} minecraft:${itemName} ${ev.type === "place" ? CONCRETE_GIVE_COUNT : 1}`);
  await sleep(800);
  const item = bot.inventory.items().find((i) => i.name === itemName);
  if (!item) {
    console.log(`[${bot.username}] ${itemName} not in inventory`);
    return null;
  }
  if (!(bot.heldItem && bot.heldItem.name === itemName)) await bot.equip(item, "hand");
  if (bot.inventory.slots[45]) await bot.unequip("off-hand");
  const hands = {
    held_item: bot.heldItem ? bot.heldItem.name : null,
    offhand_item: bot.inventory.slots[45] ? bot.inventory.slots[45].name : null,
    quick_bar_slot: bot.quickBarSlot,
  };
  console.log(`[${bot.username}] hands: ${JSON.stringify(hands)}`);
  return hands.held_item === itemName && hands.offhand_item === null ? hands : null;
}

// The interaction itself, camera untouched. Fills the timing / verification
// fields of ev: trigger_time_ms, client_done_time_ms, effect_time_ms (the
// server confirmed the new block state; polled every ~20 ms from the
// trigger, so it is at most one poll late).
async function interact(bot, rcon, ev) {
  const after = ev.type === "place" ? ev.block : "air";
  const pos = new Vec3(ev.pos[0], ev.pos[1], ev.pos[2]);
  ev.held_item = bot.heldItem ? bot.heldItem.name : null;
  ev.offhand_item = bot.inventory.slots[45] ? bot.inventory.slots[45].name : null;
  ev.trigger_time_ms = Date.now();
  const confirm = rconWaitBlock(rcon, ev.pos, after, EVENT_TIMEOUT_MS).catch(() => null);
  try {
    if (ev.type === "place") {
      const ref = bot.blockAt(pos.offset(0, -1, 0));
      await withTimeout(
        bot._placeBlockWithOptions(ref, new Vec3(0, 1, 0), { swingArm: "right", offhand: false, forceLook: "ignore" }),
        EVENT_TIMEOUT_MS,
        "place",
      );
    } else {
      const block = bot.blockAt(pos);
      if (!block || block.boundingBox === "empty") throw new Error(`nothing to break at ${pos}: ${block && block.name}`);
      await withTimeout(bot.dig(block, "ignore"), EVENT_TIMEOUT_MS, "dig");
    }
    ev.error = null;
  } catch (err) {
    ev.error = err.message;
    console.log(`[${bot.username}] ${ev.type} failed: ${err.message}`);
  }
  ev.client_done_time_ms = Date.now();
  const confirmed = await confirm;
  ev.effect_time_ms = confirmed;
  ev.server_block_after_ok = confirmed !== null;
  const b = bot.blockAt(pos);
  ev.client_block_after = b ? b.name : null;
  ev.world_changed = ev.server_block_after_ok && ev.client_block_after === after;
  console.log(
    `[${bot.username}] ${ev.type} ${ev.block} at ${ev.pos}: client=${ev.client_block_after}, server_ok=${ev.server_block_after_ok}, ` +
      `client ${ev.client_done_time_ms - ev.trigger_time_ms} ms, server ${confirmed === null ? "-" : confirmed - ev.trigger_time_ms} ms`,
  );
}

// State of the target at the end of the episode (the change must persist).
async function finalState(bot, rcon, ev) {
  const after = ev.type === "place" ? ev.block : "air";
  ev.server_block_end_ok = (await rconBlockIs(rcon, ev.pos, after)) && (await rconBlockIs(rcon, ev.base_pos, COLUMN_BASE_BLOCK));
  const b = bot.blockAt(new Vec3(ev.pos[0], ev.pos[1], ev.pos[2]));
  return b ? b.name : null;
}

// Force the recording camera onto the bot's exact look (it only re-syncs
// after a turn > 1 deg), then settle.
async function syncCamera(bot, yaw0) {
  await bot.look(yaw0 + SYNC_WIGGLE_DEG * DEG, SYNC_WIGGLE_DEG * DEG, true);
  await bot.waitForTicks(4);
  await bot.look(yaw0, 0, true);
  await bot.waitForTicks(LEAD_TICKS);
}

/**
 * Run a script stepped by the frame recorder. Step 0 is the first recorded
 * frame at or after t0; every later frame advances by one step (or by
 * round(gap / 50 ms) after a gap >= 75 ms: a frame the recorder skipped).
 * degAt(step) is the yaw offset in degrees (applied as yaw0 + sign * deg,
 * pitch 0); onTrigger fires once on the first frame at or past triggerStep.
 * Resolves after windowSteps + TAIL_FRAMES steps.
 */
async function runFrameScript(bot, { t0, windowSteps, yaw0, sign, degAt, triggerStep, onTrigger }) {
  const trace = [];
  const original = bot.getLastCameraAction;
  let anchor = null;
  let lastDeg = 0;
  let step = 0;
  let lastFrameMs = 0;
  let fired = false;
  let finish;
  const done = new Promise((resolve) => (finish = resolve));
  bot.getLastCameraAction = () => {
    try {
      const now = Date.now();
      if (anchor === null && now >= t0) {
        anchor = now;
        step = 0;
      } else if (anchor !== null) {
        const dt = now - lastFrameMs;
        step += dt >= DROP_GAP_MS ? Math.round(dt / FRAME_MS) : 1;
      }
      lastFrameMs = now;
      if (anchor !== null) {
        if (step < windowSteps + TAIL_FRAMES) {
          const deg = degAt(Math.min(step, windowSteps - 1));
          if (deg !== lastDeg) {
            lastDeg = deg;
            bot.look(yaw0 + sign * deg * DEG, 0, true).catch(() => {});
          }
          if (!fired && triggerStep !== null && step >= triggerStep) {
            fired = true;
            onTrigger();
          }
          trace.push([now, step, deg]);
        } else {
          finish();
        }
      }
    } catch (err) {
      console.log(`[${bot.username}] frame hook error: ${err.message}`);
    }
    return original();
  };
  const timeout = new Promise((resolve) =>
    setTimeout(() => resolve("timeout"), Math.max(0, t0 - Date.now()) + (windowSteps + TAIL_FRAMES) * FRAME_MS + 15000),
  );
  let how;
  try {
    how = await Promise.race([done, timeout]);
  } finally {
    bot.getLastCameraAction = original;
  }
  const inWindow = trace.filter((p) => p[1] < windowSteps);
  return {
    trace,
    anchor,
    timedOut: how === "timeout",
    windowEnd: inWindow.length ? inWindow[inWindow.length - 1][0] + FRAME_MS / 2 : null,
    recorded: inWindow.length,
  };
}

function planPhases(lens) {
  let s = 0;
  return lens.map(([name, frames, extra]) => {
    const p = { name, start_step: s, end_step: s + frames - 1, frames, ...(extra || {}) };
    s += frames;
    return p;
  });
}

function storeScript(meta, t0, res) {
  meta.t0_time_ms = t0;
  meta.window_start_time_ms = res.anchor;
  meta.window_end_time_ms = res.windowEnd;
  meta.frame_trace = res.trace;
  meta.frame_trace_format = "[hook time ms (the frame's renderTime is 0-2 ms later), script step, degrees turned]";
  meta.hook_timed_out = res.timedOut;
}

// ---------------------------------------------------------------------------
// EP: worldModRevisitEval
// ---------------------------------------------------------------------------

function revisitSpec(instanceId, episodeNum, isFarBot) {
  const k = (((3 * instanceId + episodeNum) % 6) + 6) % 6;
  return {
    variant_index: k,
    hold_frames: HOLD_FRAMES[k % 3],
    interaction: Math.floor(k / 3) % 2 === 0 ? "place" : "break",
    direction: DIRECTIONS[(k + (isFarBot ? 1 : 0)) % 2],
  };
}

function revisitPhasePlan(spec) {
  return planPhases([
    ["static_pre", STATIC_PRE_FRAMES],
    ["interact", INTERACT_FRAMES[spec.interaction]],
    ["static_mid", POST_INTERACT_STATIC_FRAMES],
    ["turn_out", TURN_FRAMES],
    ["hold", spec.hold_frames],
    ["turn_back", TURN_FRAMES],
    ["static_post", STATIC_POST_FRAMES],
  ]);
}

function revisitDeg(spec, s) {
  const t1 = STATIC_PRE_FRAMES + INTERACT_FRAMES[spec.interaction] + POST_INTERACT_STATIC_FRAMES;
  const r = t1 + TURN_FRAMES + spec.hold_frames;
  if (s < t1) return 0;
  if (s < r) return Math.min(DEG_PER_FRAME * (s - t1 + 1), THETA_DEG);
  return THETA_DEG - Math.min(DEG_PER_FRAME * (s - r + 1), THETA_DEG);
}

async function pickRevisitScene(bot, rng, signedThetaRad) {
  const base = bot.entity.position.floored();
  const columns = [];
  for (let dx = -SCENE_SEARCH_RADIUS; dx <= SCENE_SEARCH_RADIUS; dx++) {
    for (let dz = -SCENE_SEARCH_RADIUS; dz <= SCENE_SEARCH_RADIUS; dz++) columns.push({ dx, dz, d2: dx * dx + dz * dz });
  }
  columns.sort((a, b) => a.d2 - b.d2);
  const deadline = Date.now() + SCENE_SEARCH_BUDGET_MS;
  const rejected = { spot: 0, column: 0, los: 0, view: 0, away_view: 0, near_occluder: 0 };
  let candidates = 0;
  for (const col of columns) {
    if (Date.now() > deadline) break;
    const x = base.x + col.dx + 0.5;
    const z = base.z + col.dz + 0.5;
    for (let dy = 2; dy >= -2; dy--) {
      const y = base.y + dy;
      if (!isStandable(bot, x, y, z)) continue;
      const pos = new Vec3(x, y, z);
      if (!isSpotClear(bot, pos, SCENE_CLEAR_RADIUS)) {
        rejected.spot++;
        break;
      }
      for (const off of shuffled(ACTOR_OFFSETS, rng)) {
        candidates++;
        const tx = Math.floor(x) + off.dx;
        const tz = Math.floor(z) + off.dz;
        if (!columnOk(bot, tx, y, tz)) {
          rejected.column++;
          continue;
        }
        const ev = makeEvent(null, null, tx, y, tz);
        if (!targetVisibleFrom(bot, pos, ev)) {
          rejected.los++;
          continue;
        }
        const yaw = yawTo(pos, ev);
        if (!isViewGood(bot, pos, yaw)) {
          rejected.view++;
          continue;
        }
        if (!isViewGood(bot, pos, yaw + signedThetaRad)) {
          rejected.away_view++;
          continue;
        }
        const occluders = [nearOccluderFraction(bot, pos, yaw), nearOccluderFraction(bot, pos, yaw + signedThetaRad)];
        if (Math.max(...occluders) > NEAR_OCCLUDER_MAX_FRACTION) {
          rejected.near_occluder++;
          continue;
        }
        return { pos, yaw, ev, dist: off.h, candidates, occluders };
      }
      break;
    }
    await sleep(0);
  }
  console.log(`[${bot.username}] scene search: nothing among ${candidates} candidates, rejected by ${JSON.stringify(rejected)}`);
  return null;
}

function getOnRevisitPhaseFn(bot, rcon, sharedBotRng, coordinator, episodeNum, episodeInstance, args) {
  return async () => {
    coordinator.sendToOtherBot("worldModRevisitPhase", bot.entity.position.clone(), episodeNum, "worldModRevisitPhase beginning");
    const plan = episodeInstance._plan;
    const meta = episodeInstance._episodeMetadata;
    const stopNow = () => {
      coordinator.onceEvent(
        "stopPhase",
        episodeNum,
        episodeInstance.getOnStopPhaseFn(bot, rcon, sharedBotRng, coordinator, args.other_bot_name, episodeNum, args),
      );
      coordinator.sendToOtherBot("stopPhase", bot.entity.position.clone(), episodeNum, "worldModRevisitPhase end");
    };
    const spec = episodeInstance._spec;
    const phases = revisitPhasePlan(spec);
    const windowSteps = phases[phases.length - 1].end_step + 1;
    if (!plan) {
      // Keep the other bot's (independent) recording whole: idle for the
      // length of the script before stopping.
      console.log(`[${bot.username}] no scene, idling for the script length`);
      await sleep(3000 + (windowSteps + TAIL_FRAMES) * FRAME_MS);
      stopNow();
      return;
    }
    const ev = plan.ev;
    try {
      stopAll(bot);
      await syncCamera(bot, plan.yaw);
      const start = bot.entity.position.clone();
      const t0 = Date.now() + START_DELAY_MS;
      let interaction = null;
      const res = await runFrameScript(bot, {
        t0,
        windowSteps,
        yaw0: plan.yaw,
        sign: spec.direction === "left" ? 1 : -1,
        degAt: (s) => revisitDeg(spec, s),
        triggerStep: STATIC_PRE_FRAMES,
        onTrigger: () => {
          interaction = interact(bot, rcon, ev);
        },
      });
      if (interaction) await interaction;
      else ev.error = "interaction never triggered";
      ev.client_block_end = await finalState(bot, rcon, ev);
      storeScript(meta, t0, res);
      meta.event = ev;
      meta.events = [ev];
      meta.drift_blocks = bot.entity.position.distanceTo(start);
      console.log(
        `[${bot.username}] revisit done: ${spec.interaction} ${spec.direction} H ${spec.hold_frames}, ${res.recorded}/${windowSteps} window frames, ` +
          `changed=${ev.world_changed}, end_ok=${ev.server_block_end_ok}, drift ${meta.drift_blocks.toFixed(4)}`,
      );
    } finally {
      stopAll(bot);
    }
    stopNow();
  };
}

/**
 * worldModRevisitEval ("EP", long-term interaction fidelity, single agent).
 * The bot faces the target cell (see the header), stands still 10 frames,
 * performs its one interaction, stands still 15, yaws out 120 deg at 3 deg
 * per frame, holds off-screen for H = 0 / 40 / 160 frames, yaws back and
 * stands still 40. Pitch 0, no translation.
 *
 * Both bots of a world record their own, independent episode far out of
 * each other's sight (as in trajRevisitEval). H and place / break follow
 * k = (3 * instance + episode) % 6 and are the same for both bots (equal
 * script length); the turn direction alternates and is opposite for the two.
 *
 * episode_info at record time: eval_kind "EP", hold_frames, interaction,
 * direction, target_cell, phase_plan (script steps), frame_trace, event.
 * postprocess/world_mod_offscreen_quality.py turns them into `phases`
 * (start_frame / end_frame) and event.effect_frame.
 * @extends BaseEpisode
 */
class WorldModRevisitEvalEpisode extends BaseEpisode {
  static WORKS_IN_NON_FLAT_WORLD = true;
  static INIT_MIN_BOTS_DISTANCE = 20;
  static INIT_MAX_BOTS_DISTANCE = 28;

  async setupEpisode(bot, rcon, sharedBotRng, coordinator, episodeNum, args, botPosition, otherBotPosition) {
    const instance = args.instance_id ?? 0;
    const seed = `worldModRevisitEval_${REVISIT_BASE_SEED}_${instance}_${episodeNum}_${bot.username}`;
    const rng = seedrandom(seed);
    const isFarBot = bot.username > args.other_bot_name;
    const spec = revisitSpec(instance, episodeNum, isFarBot);
    this._spec = spec;
    this._plan = null;
    this._builtEvent = null;
    const signedThetaDeg = spec.direction === "left" ? THETA_DEG : -THETA_DEG;
    const block = BLOCK_TYPES[Math.floor(rng() * BLOCK_TYPES.length)];
    console.log(`[${bot.username}] Episode ${episodeNum}: EP ${spec.interaction} ${block}, ${spec.direction} ${THETA_DEG} deg, hold ${spec.hold_frames}`);
    this._episodeMetadata = {
      eval_kind: "EP",
      world_mod_revisit_seed: seed,
      ...spec,
      block,
      theta_deg: THETA_DEG,
      signed_theta_deg: signedThetaDeg,
      deg_per_frame: DEG_PER_FRAME,
      horizontal_fov_deg: HFOV_DEG,
      vertical_fov_deg: 70,
      fps: 20,
      tail_frames: TAIL_FRAMES,
      scene_check_passed: false,
      phase_plan: revisitPhasePlan(spec),
      suggested_window: "window_start_frame .. window_end_frame (static_pre .. static_post); frames before it are the static lead, after it a static tail",
    };
    this._evalMetadata = { bots_chosen: [bot.username], interaction: spec.interaction, direction: spec.direction, hold_frames: spec.hold_frames };
    const fail = () => ({ botPositionNew: botPosition, otherBotPositionNew: otherBotPosition });

    await quietNotificationsOnce(bot, rcon);

    let origin = botPosition;
    if (isFarBot) {
      const farRng = seedrandom(`${seed}_far`);
      const ang0 = farRng() * 2 * Math.PI;
      for (let i = 0; i < FAR_MAX_TRIES; i++) {
        const ang = ang0 + (i * 2 * Math.PI) / FAR_MAX_TRIES;
        const fx = Math.floor(botPosition.x + FAR_OFFSET_BLOCKS * Math.cos(ang));
        const fz = Math.floor(botPosition.z + FAR_OFFSET_BLOCKS * Math.sin(ang));
        const res = await rconLog(bot, rcon, `spreadplayers ${fx} ${fz} 0 ${FAR_SPREAD_RADIUS} false ${bot.username}`);
        if (/^Spread/.test(res)) break;
      }
      await sleep(FAR_SETTLE_MS);
      origin = bot.entity.position.clone();
      if (Math.hypot(origin.x - botPosition.x, origin.z - botPosition.z) < FAR_OFFSET_BLOCKS - 2 * FAR_SPREAD_RADIUS - 8) {
        console.log(`[${bot.username}] could not leave the other bot's surroundings`);
        return fail();
      }
    }

    let scene = null;
    let attempts = 0;
    while (scene === null && attempts < SCENE_MAX_ATTEMPTS) {
      if (attempts > 0) {
        const range = Math.min(SCENE_HOP_RANGE * 2 ** (attempts - 1), SCENE_HOP_MAX_RANGE);
        await rconLog(bot, rcon, `spreadplayers ${Math.floor(origin.x)} ${Math.floor(origin.z)} 0 ${range} false ${bot.username}`);
        await sleep(SCENE_HOP_SETTLE_MS);
      }
      attempts++;
      scene = await pickRevisitScene(bot, rng, signedThetaDeg * DEG);
    }
    this._episodeMetadata.scene_search_attempts = attempts;
    if (scene === null) return fail();
    const { pos } = scene;
    const ev = { ...scene.ev, type: spec.interaction, block, actor: bot.username };
    await rconLog(bot, rcon, `tp ${bot.username} ${pos.x} ${pos.y} ${pos.z}`);
    await sleep(1500);
    if (bot.entity.position.distanceTo(pos) > 0.3) {
      console.log(`[${bot.username}] not at the scene start (${bot.entity.position})`);
      return fail();
    }
    await prepareWorld(bot, rcon, pos);
    if (!(await buildColumn(bot, rcon, ev))) return fail();
    const hands = await armActor(bot, rcon, ev);
    if (!hands) return fail();
    console.log(`[${bot.username}] scene (${pos.x}, ${pos.y}, ${pos.z}) yaw ${(scene.yaw / DEG).toFixed(1)} deg, target ${ev.pos} at ${scene.dist.toFixed(2)}, ${attempts} attempt(s)`);

    this._plan = { yaw: scene.yaw, ev };
    this._builtEvent = ev;
    Object.assign(this._episodeMetadata, {
      scene_check_passed: true,
      target_cell: ev.pos,
      pedestal_cell: ev.base_pos,
      pedestal_block: COLUMN_BASE_BLOCK,
      target_distance: scene.dist,
      near_occluder_fraction: { start_view: scene.occluders[0], turned_out_view: scene.occluders[1], max_allowed: NEAR_OCCLUDER_MAX_FRACTION, range_blocks: NEAR_OCCLUDER_RANGE },
      held_item: hands.held_item,
      start_pos: [pos.x, pos.y, pos.z],
      start_yaw: scene.yaw,
      start_pitch: 0,
      event: ev,
      events: [ev],
    });
    const ahead = new Vec3(pos.x - Math.sin(scene.yaw) * 16, pos.y, pos.z - Math.cos(scene.yaw) * 16);
    return { botPositionNew: pos, otherBotPositionNew: ahead };
  }

  async entryPoint(bot, rcon, sharedBotRng, coordinator, iterationID, episodeNum, args) {
    coordinator.onceEvent("worldModRevisitPhase", episodeNum, getOnRevisitPhaseFn(bot, rcon, sharedBotRng, coordinator, episodeNum, this, args));
    coordinator.sendToOtherBot("worldModRevisitPhase", bot.entity.position.clone(), episodeNum, "entryPoint end");
  }

  async tearDownEpisode(bot, rcon, sharedBotRng, coordinator, episodeNum, args) {
    for (const cmd of ["gamerule doMobSpawning true", "gamerule doMobLoot true", "gamerule doTileDrops true"]) await rconLog(bot, rcon, cmd);
    if (this._builtEvent) {
      for (const p of [this._builtEvent.pos, this._builtEvent.base_pos]) {
        try {
          await rconSetBlock(rcon, p, "air");
        } catch (err) {
          console.log(`[${bot.username}] cleanup setblock failed: ${err.message}`);
        }
      }
    }
  }
}

const modScriptUtils = {
  rconLog,
  nearOccluderFraction,
  isViewOpen,
  NEAR_OCCLUDER_MAX_FRACTION,
  NEAR_OCCLUDER_RANGE,
  isSpotClear,
  columnOk,
  makeEvent,
  yawTo,
  targetVisibleFrom,
  buildColumn,
  prepareWorld,
  armActor,
  interact,
  finalState,
  syncCamera,
  runFrameScript,
  planPhases,
  storeScript,
  COLUMN_BASE_BLOCK,
  INTERACT_FRAMES,
  DEG_PER_FRAME,
  STATIC_PRE_FRAMES,
  TAIL_FRAMES,
  FRAME_MS,
  HOLD_FRAMES,
  HFOV_DEG,
  ACTOR_OFFSETS,
};

module.exports = { WorldModRevisitEvalEpisode, modScriptUtils };

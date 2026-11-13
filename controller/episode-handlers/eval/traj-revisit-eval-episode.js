const seedrandom = require("seedrandom");
const { Vec3 } = require("vec3");

const { stopAll } = require("../../primitives/movement");
const { sleep } = require("../../utils/helpers");
const { BaseEpisode } = require("../base-episode");
const { sceneUtils } = require("./traj-segments-eval-episode");
const { worldModUtils } = require("./world-mod-eval-episode");

const { isStandable, isOpenColumn, isViewGood } = sceneUtils;
const { quietNotificationsOnce, MOB_CLEAR_RADIUS, DEG } = worldModUtils;

// Fixed base seed; the per-recording seed is derived from (base seed,
// instance id, episode number, bot name).
const REVISIT_BASE_SEED = 20261113;

// The script (physics ticks; the recording runs at 20 fps, 1 tick = 1 frame):
// static, yaw out by THETA_DEG at OMEGA_DEG_PER_SEC, hold off-screen for H
// ticks, yaw back at the same rate, static. No translation, pitch 0.
//
// THETA_DEG must exceed the horizontal FOV (102.4 deg at 1280x720 with the
// client's vertical FOV 70) for the start picture to leave the view
// completely; 120 deg leaves a 17.6 deg margin and takes exactly 40 ticks.
const THETA_DEG = 120;
const OMEGA_DEG_PER_SEC = 60;
const HFOV_DEG = 102.45;
const HOLD_TIERS = [
  { name: "zero", ticks: 0 },
  { name: "mid", ticks: 40 },
  { name: "long", ticks: 160 },
];
const DIRECTIONS = ["left", "right"]; // left = counter-clockwise = yaw grows
const PRE_STATIC_TICKS = 30;
const POST_STATIC_TICKS = 40; // >= 30 required; slack for the video lag
const LEAD_TICKS = 20; // settle before the window starts
const SYNC_WIGGLE_DEG = 6;
const TAIL_TICKS = 6;

// Scene search (same scheme as trajSegmentsEval: nearest level, clear spot
// with a good view, random hops when the neighbourhood has none).
const SCENE_SEARCH_RADIUS = 8;
const SCENE_YAW_CANDIDATES = 12;
const SCENE_SEARCH_BUDGET_MS = 4000;
const SCENE_MAX_ATTEMPTS = 8;
const SCENE_HOP_RANGE = 16;
const SCENE_HOP_MAX_RANGE = 64;
const SCENE_HOP_SETTLE_MS = 4000;
const SCENE_CLEAR_RADIUS = 2;

// The two bots of a world record independently and must never see each
// other (players are tracked up to 128 blocks): the second bot (higher name)
// is moved FAR_OFFSET_BLOCKS away before its scene search; with both hop
// ranges capped at SCENE_HOP_MAX_RANGE they stay > 200 blocks apart.
const FAR_OFFSET_BLOCKS = 400;
const FAR_SPREAD_RADIUS = 16;
const FAR_SETTLE_MS = 6000;
const FAR_MAX_TRIES = 8;

function isSpotClear(bot, pos) {
  for (let dx = -SCENE_CLEAR_RADIUS; dx <= SCENE_CLEAR_RADIUS; dx++) {
    for (let dz = -SCENE_CLEAR_RADIUS; dz <= SCENE_CLEAR_RADIUS; dz++) {
      if (!isOpenColumn(bot, pos.x + dx, pos.y, pos.z + dz)) return false;
    }
  }
  return true;
}

/**
 * Search the bot's neighbourhood for a start pose: a level, clear block
 * centre and a yaw whose view is open with landmarks in sight, both at the
 * start and at the turned-out direction (yaw + signedTheta).
 * @returns {{pos: Vec3, yaw: number, candidates: number}|null}
 */
async function pickScene(bot, rng, signedThetaRad) {
  const base = bot.entity.position.floored();
  const columns = [];
  for (let dx = -SCENE_SEARCH_RADIUS; dx <= SCENE_SEARCH_RADIUS; dx++) {
    for (let dz = -SCENE_SEARCH_RADIUS; dz <= SCENE_SEARCH_RADIUS; dz++) {
      columns.push({ dx, dz, d2: dx * dx + dz * dz });
    }
  }
  columns.sort((a, b) => a.d2 - b.d2);
  const yaws = [];
  for (let k = 0; k < SCENE_YAW_CANDIDATES; k++) {
    yaws.push(((k + rng()) * 2 * Math.PI) / SCENE_YAW_CANDIDATES);
  }
  for (let k = yaws.length - 1; k > 0; k--) {
    const j = Math.floor(rng() * (k + 1));
    [yaws[k], yaws[j]] = [yaws[j], yaws[k]];
  }

  const deadline = Date.now() + SCENE_SEARCH_BUDGET_MS;
  const rejected = { spot: 0, view: 0, away_view: 0 };
  let candidates = 0;
  for (const col of columns) {
    if (Date.now() > deadline) break;
    const x = base.x + col.dx + 0.5;
    const z = base.z + col.dz + 0.5;
    for (let dy = 2; dy >= -2; dy--) {
      const y = base.y + dy;
      if (!isStandable(bot, x, y, z)) continue;
      const pos = new Vec3(x, y, z);
      if (!isSpotClear(bot, pos)) {
        rejected.spot++;
        break;
      }
      for (const yaw of yaws) {
        candidates++;
        if (!isViewGood(bot, pos, yaw)) {
          rejected.view++;
          continue;
        }
        if (!isViewGood(bot, pos, yaw + signedThetaRad)) {
          rejected.away_view++;
          continue;
        }
        return { pos, yaw, candidates };
      }
      break; // the first standable y of a column is the surface
    }
    await sleep(0);
  }
  console.log(
    `[${bot.username}] scene search: no start pose among ${candidates} candidates, rejected by ${JSON.stringify(rejected)}`,
  );
  return null;
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

// Rotate yaw by deltaRad at a constant rate, one forced look per tick, ending
// exactly on the target. `look` holds the unwrapped yaw/pitch and is updated
// in place. Returns the ticks spent.
async function stepYaw(bot, look, deltaRad, omegaDegPerSec) {
  const n = Math.max(1, Math.round(Math.abs(deltaRad) / ((omegaDegPerSec * DEG) / 20)));
  const y0 = look.yaw;
  for (let i = 1; i <= n; i++) {
    look.yaw = y0 + (deltaRad * i) / n;
    await bot.look(look.yaw, look.pitch, true);
    await bot.waitForTicks(1);
  }
  return n;
}

function getOnRevisitPhaseFn(bot, rcon, sharedBotRng, coordinator, episodeNum, episodeInstance, args) {
  return async () => {
    coordinator.sendToOtherBot("revisitPhase", bot.entity.position.clone(), episodeNum, "revisitPhase beginning");
    const plan = episodeInstance._plan;
    const meta = episodeInstance._episodeMetadata;
    const phases = meta.phases;
    const phase = (name) => phases.find((p) => p.name === name);
    const run = async (name, fn) => {
      const p = phase(name);
      p.start_time_ms = Date.now();
      p.ticks = await fn();
      p.end_time_ms = Date.now();
    };

    try {
      stopAll(bot);
      const look = { yaw: plan.yaw, pitch: 0 };
      // The recording camera is only re-synced to the bot when the bot moved
      // > 0.05 blocks or turned > 1 deg since the last sync, so after the
      // setup's aim it can sit up to 1 deg off the bot's yaw, while the last
      // 3 deg step of the return syncs it exactly. Force an exact sync now:
      // look away by several degrees on both axes, then back.
      await bot.look(look.yaw + SYNC_WIGGLE_DEG * DEG, SYNC_WIGGLE_DEG * DEG, true);
      await bot.waitForTicks(4);
      await bot.look(look.yaw, look.pitch, true);
      await bot.waitForTicks(LEAD_TICKS);
      const posBefore = bot.entity.position.clone();
      const yaw0 = look.yaw;
      const delta = plan.signedThetaDeg * DEG;

      meta.window_start_time_ms = Date.now();
      await run("static_pre", async () => {
        await bot.waitForTicks(PRE_STATIC_TICKS);
        return PRE_STATIC_TICKS;
      });
      await run("turn_out", () => stepYaw(bot, look, delta, OMEGA_DEG_PER_SEC));
      await run("hold", async () => {
        if (plan.holdTicks > 0) await bot.waitForTicks(plan.holdTicks);
        return plan.holdTicks;
      });
      await run("turn_back", async () => {
        const n = await stepYaw(bot, look, -delta, OMEGA_DEG_PER_SEC);
        look.yaw = yaw0;
        await bot.look(look.yaw, look.pitch, true);
        return n;
      });
      await run("static_post", async () => {
        await bot.waitForTicks(POST_STATIC_TICKS);
        return POST_STATIC_TICKS;
      });
      meta.window_end_time_ms = Date.now();

      const posAfter = bot.entity.position.clone();
      meta.position_drift_blocks = posAfter.distanceTo(posBefore);
      meta.end_yaw = bot.entity.yaw;
      meta.end_pitch = bot.entity.pitch;
      console.log(
        `[${bot.username}] revisit done: ${plan.direction} ${THETA_DEG} deg, hold ${plan.holdTicks} ticks, ` +
          `drift ${meta.position_drift_blocks.toFixed(4)} blocks`,
      );
      await bot.waitForTicks(TAIL_TICKS);
    } finally {
      stopAll(bot);
    }

    coordinator.onceEvent(
      "stopPhase",
      episodeNum,
      episodeInstance.getOnStopPhaseFn(bot, rcon, sharedBotRng, coordinator, args.other_bot_name, episodeNum, args),
    );
    coordinator.sendToOtherBot("stopPhase", bot.entity.position.clone(), episodeNum, "revisitPhase end");
  };
}

/**
 * Single-agent revisit eval episode (long-term consistency): the bot stands
 * still, yaws out by THETA_DEG (more than the horizontal FOV, so the start
 * picture leaves the view completely) at a constant OMEGA_DEG_PER_SEC, holds
 * off-screen for H ticks (HOLD_TIERS), yaws back to the exact start direction
 * at the same rate and stands still again. No translation, pitch 0.
 *
 * Both bots of a world record their own, independent episode far out of each
 * other's sight. The hold tier cycles with (instance id + episode number) and
 * is the same for both bots; the turn direction alternates and is opposite
 * for the two bots, so tiers and directions stay balanced.
 *
 * episode_info carries theta_deg, signed_theta_deg, direction, hold_tier,
 * hold_frames and `phases` (static_pre, turn_out, hold, turn_back,
 * static_post) with their wall-clock windows; postprocess/traj_segments.py
 * turns them into start_frame / end_frame when the dataset is flattened.
 * @extends BaseEpisode
 */
class TrajRevisitEvalEpisode extends BaseEpisode {
  static WORKS_IN_NON_FLAT_WORLD = true;
  static INIT_MIN_BOTS_DISTANCE = 20;
  static INIT_MAX_BOTS_DISTANCE = 28;

  async setupEpisode(bot, rcon, sharedBotRng, coordinator, episodeNum, args, botPosition, otherBotPosition) {
    const instance = args.instance_id ?? 0;
    const seed = `trajRevisitEval_${REVISIT_BASE_SEED}_${instance}_${episodeNum}_${bot.username}`;
    const rng = seedrandom(seed);
    const isFarBot = bot.username > args.other_bot_name;
    const k = (instance + episodeNum) % (HOLD_TIERS.length * DIRECTIONS.length);
    const tier = HOLD_TIERS[k % HOLD_TIERS.length];
    const direction = DIRECTIONS[(Math.floor(k / HOLD_TIERS.length) + (isFarBot ? 1 : 0)) % DIRECTIONS.length];
    const signedThetaDeg = direction === "left" ? THETA_DEG : -THETA_DEG;
    console.log(
      `[${bot.username}] Episode ${episodeNum}: revisit ${direction} ${THETA_DEG} deg, hold ${tier.name} (${tier.ticks} ticks)`,
    );

    await quietNotificationsOnce(bot, rcon);

    // Second bot: leave the first bot's surroundings for good.
    let origin = botPosition;
    let farOk = true;
    if (isFarBot) {
      // spreadplayers refuses water; try other directions until it lands.
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
      farOk = Math.hypot(origin.x - botPosition.x, origin.z - botPosition.z) > FAR_OFFSET_BLOCKS - 2 * FAR_SPREAD_RADIUS - 8;
      if (!farOk) console.log(`[${bot.username}] could not leave the other bot's surroundings, episode will be flagged`);
    }

    let scene = null;
    let attempts = 0;
    while (farOk && scene === null && attempts < SCENE_MAX_ATTEMPTS) {
      if (attempts > 0) {
        const range = Math.min(SCENE_HOP_RANGE * 2 ** (attempts - 1), SCENE_HOP_MAX_RANGE);
        await rconLog(bot, rcon, `spreadplayers ${Math.floor(origin.x)} ${Math.floor(origin.z)} 0 ${range} false ${bot.username}`);
        await sleep(SCENE_HOP_SETTLE_MS);
      }
      attempts++;
      scene = await pickScene(bot, rng, signedThetaDeg * DEG);
    }
    let moved = false;
    if (scene !== null) {
      const { pos } = scene;
      await rconLog(bot, rcon, `tp ${bot.username} ${pos.x} ${pos.y} ${pos.z}`);
      await sleep(1500);
      moved = bot.entity.position.distanceTo(pos) < 0.3;
      console.log(
        `[${bot.username}] scene start (${pos.x}, ${pos.y}, ${pos.z}) yaw=${(scene.yaw / DEG).toFixed(1)} deg ` +
          `after ${scene.candidates} candidates, ${attempts} attempt(s), in place=${moved}`,
      );
    }

    // Empty hands, no mobs (and no loot from the cleared ones), no weather.
    const p = bot.entity.position;
    const kill = `execute positioned ${Math.floor(p.x)} ${Math.floor(p.y)} ${Math.floor(p.z)} run kill @e[type=!minecraft:player,distance=..${MOB_CLEAR_RADIUS}]`;
    for (const cmd of [`clear ${bot.username}`, "gamerule doMobSpawning false", "gamerule doMobLoot false", "weather clear 1000000", kill]) {
      await rconLog(bot, rcon, cmd);
    }
    await sleep(500);
    await rconLog(bot, rcon, kill); // anything the first pass dropped

    this._plan = moved
      ? { yaw: scene.yaw, signedThetaDeg, direction, holdTicks: tier.ticks }
      : { yaw: bot.entity.yaw, signedThetaDeg, direction, holdTicks: tier.ticks };
    this._episodeMetadata = {
      traj_kind: "revisit",
      revisit_seed: seed,
      theta_deg: THETA_DEG,
      signed_theta_deg: signedThetaDeg,
      direction,
      omega_deg_per_sec: OMEGA_DEG_PER_SEC,
      hold_tier: tier.name,
      hold_frames: tier.ticks,
      pre_static_frames: PRE_STATIC_TICKS,
      post_static_frames: POST_STATIC_TICKS,
      turn_frames: Math.round(THETA_DEG / (OMEGA_DEG_PER_SEC / 20)),
      horizontal_fov_deg: HFOV_DEG,
      vertical_fov_deg: 70,
      fps: 20,
      scene_check_passed: moved,
      scene_search_attempts: attempts,
      start_pos: moved ? [scene.pos.x, scene.pos.y, scene.pos.z] : null,
      start_yaw: moved ? scene.yaw : null,
      start_pitch: 0,
      phases: ["static_pre", "turn_out", "hold", "turn_back", "static_post"].map((name) => ({ name })),
      suggested_window: "window_start_frame .. window_end_frame (static_pre .. static_post)",
    };
    this._evalMetadata = {
      bots_chosen: [bot.username],
      theta_deg: THETA_DEG,
      direction,
      hold_frames: tier.ticks,
    };

    if (!moved) {
      return { botPositionNew: botPosition, otherBotPositionNew: otherBotPosition };
    }
    // The caller turns the bot towards otherBotPositionNew before recording
    // starts; a point straight ahead at the bot's own height puts the camera
    // on the chosen yaw with pitch 0.
    const ahead = new Vec3(scene.pos.x - Math.sin(scene.yaw) * 16, scene.pos.y, scene.pos.z - Math.cos(scene.yaw) * 16);
    return { botPositionNew: scene.pos, otherBotPositionNew: ahead };
  }

  async entryPoint(bot, rcon, sharedBotRng, coordinator, iterationID, episodeNum, args) {
    coordinator.onceEvent("revisitPhase", episodeNum, getOnRevisitPhaseFn(bot, rcon, sharedBotRng, coordinator, episodeNum, this, args));
    coordinator.sendToOtherBot("revisitPhase", bot.entity.position.clone(), episodeNum, "entryPoint end");
  }

  async tearDownEpisode(bot, rcon, sharedBotRng, coordinator, episodeNum, args) {
    for (const cmd of ["gamerule doMobSpawning true", "gamerule doMobLoot true"]) {
      await rconLog(bot, rcon, cmd);
    }
  }
}

module.exports = { TrajRevisitEvalEpisode };

const seedrandom = require("seedrandom");
const { Vec3 } = require("vec3");

const { stopAll } = require("../../primitives/movement");
const { decidePrimaryBot } = require("../../utils/coordination");
const { sleep } = require("../../utils/helpers");
const { BaseEpisode } = require("../base-episode");
const { sceneUtils } = require("./traj-segments-eval-episode");
const { worldModUtils } = require("./world-mod-eval-episode");

const { blockAtXYZ, isStandable, isOpenColumn, isViewGood, EYE_HEIGHT } = sceneUtils;
const { eyeOf, anglesTo, quietNotificationsOnce, rendezvous, lookState, lookExact, shuffled, MOB_CLEAR_RADIUS, DEG } =
  worldModUtils;

// Fixed base seed; the per-episode seed is derived from (base seed, instance
// id, episode number) so both bots agree on the plan.
const ACTOR_MOTION_BASE_SEED = 20261103;

// One atomic body action per episode, cycled with (instance id + episode
// number) so a 2-world x 6-episode trial holds two of each.
const ACTIONS = ["forward", "back", "left", "right", "jump", "sneak"];
const KEY_OF = { forward: "forward", back: "back", left: "left", right: "right", jump: "jump", sneak: "sneak" };

// Timing (physics ticks; the recording runs at ~20 fps so 1 tick ~ 1 frame).
// Both bots stand still for PRE_HOLD_TICKS, the actor performs the action
// (a walk key held WALK_TICKS plus a few ticks of slide, one jump of ~12
// airborne ticks, or sneak held SNEAK_TICKS), then both keep still for at
// least POST_MIN_TICKS, the window being padded to WINDOW_TARGET_TICKS.
const PRE_HOLD_TICKS = 20;
const WALK_TICKS = 14; // ~3 blocks of plain walking (no sprint)
const SNEAK_TICKS = 20;
const JUMP_MAX_TICKS = 30;
const SLIDE_MAX_TICKS = 12;
const SLIDE_STOP_EPS = 1e-3; // blocks per tick
const POST_MIN_TICKS = 40;
const WINDOW_TARGET_TICKS = 112;

// Geometry. The observer B stands OBSERVER_DIST_RANGE blocks (uniform per
// episode) from the actor A on the same level, both face each other (A's
// "forward" is towards B). At 6-7 blocks A is 138-163 px tall in B's 720p
// frame and a forward walk still ends fully inside the frame (trial001 tried
// 6-9). The walk corridor (WALK_BLOCKS + margin) must be flat and free for
// the action's key direction.
const OBSERVER_DIST_RANGE = [6, 7];
const WALK_BLOCKS = 3.5;
const CORRIDOR_MARGIN_BLOCKS = 1.5;
const BODY_HEIGHT = 1.8;
const BODY_CENTRE = 0.9;

// Scene search (see pickScene).
const SCENE_SEARCH_RADIUS = 8;
const SCENE_YAW_CANDIDATES = 12;
const SCENE_SEARCH_BUDGET_MS = 5000;
const SCENE_MAX_ATTEMPTS = 10;
const SCENE_HOP_RANGE = 24;
const SCENE_HOP_MAX_RANGE = 320;
const SCENE_HOP_SETTLE_MS = 4000;
const HOP_SPREAD_RADIUS = 24;
const FLAT_BIOMES = ["plains", "savanna", "desert", "sunflower_plains", "meadow"];
const SPOT_CLEAR_RADIUS = 1;
const OBSERVER_CHUNK_SETTLE_MS = 2500;

// ---------------------------------------------------------------------------
// Geometry helpers (mineflayer convention: yaw 0 = north = -z, yaw grows
// counter-clockwise, pitch positive = up)
// ---------------------------------------------------------------------------

// Horizontal unit vector of a movement key for a given yaw.
function keyDirection(key, yaw) {
  const offset = { forward: 0, left: 90, back: 180, right: -90 }[key];
  const a = yaw + offset * DEG;
  return { x: -Math.sin(a), z: -Math.cos(a) };
}

function bodyCentreOf(pos) {
  return { x: pos.x, y: pos.y + BODY_CENTRE, z: pos.z };
}

function isSpotClear(bot, pos) {
  for (let dx = -SPOT_CLEAR_RADIUS; dx <= SPOT_CLEAR_RADIUS; dx++) {
    for (let dz = -SPOT_CLEAR_RADIUS; dz <= SPOT_CLEAR_RADIUS; dz++) {
      if (!isOpenColumn(bot, pos.x + dx, pos.y, pos.z + dz)) return false;
    }
  }
  return true;
}

// The strip from `from` along `dir` for `length` blocks is level, standable
// ground (same y) with open columns a little to each side.
function isCorridorFree(bot, from, dir, length) {
  const side = { x: -dir.z, z: dir.x };
  for (let t = 0; t <= length; t += 0.5) {
    const cx = from.x + dir.x * t;
    const cz = from.z + dir.z * t;
    for (const w of [-0.45, 0, 0.45]) {
      if (!isStandable(bot, cx + side.x * w, from.y, cz + side.z * w)) return false;
    }
    for (const w of [-1.3, 1.3]) {
      if (!isOpenColumn(bot, cx + side.x * w, from.y, cz + side.z * w)) return false;
    }
  }
  return true;
}

// True when the segment from `from` to `to` crosses a collidable block.
function segmentBlocked(bot, from, to) {
  const d = Math.hypot(to.x - from.x, to.y - from.y, to.z - from.z);
  const n = Math.max(1, Math.ceil(d / 0.1));
  for (let i = 0; i <= n; i++) {
    const t = i / n;
    const block = blockAtXYZ(bot, from.x + (to.x - from.x) * t, from.y + (to.y - from.y) * t, from.z + (to.z - from.z) * t);
    if (block === null) return true; // unloaded: assume blocked
    if (block.boundingBox !== "empty") return true;
  }
  return false;
}

// Line of sight from an eye to a bot standing at `pos` (feet): head, body
// centre and feet of its column must be reachable.
function bodyVisibleFrom(bot, eye, pos) {
  return [0.2, BODY_CENTRE, BODY_HEIGHT - 0.1].every((h) => !segmentBlocked(bot, eye, { x: pos.x, y: pos.y + h, z: pos.z }));
}

// ---------------------------------------------------------------------------
// Scene selection
// ---------------------------------------------------------------------------

/**
 * Search the actor's surroundings for a start pose: A on level, clear ground
 * at a block centre facing B `dist` blocks away on the same level, the strip
 * between them and the walk corridor of the action free and flat, both views
 * open with landmarks in sight.
 * @returns {{actorPos, observerPos, yawA, yawB, candidates, rejected}|null}
 */
async function pickScene(bot, rng, action, dist) {
  const base = bot.entity.position.floored();
  const columns = [];
  for (let dx = -SCENE_SEARCH_RADIUS; dx <= SCENE_SEARCH_RADIUS; dx++) {
    for (let dz = -SCENE_SEARCH_RADIUS; dz <= SCENE_SEARCH_RADIUS; dz++) {
      columns.push({ dx, dz, d2: dx * dx + dz * dz });
    }
  }
  columns.sort((a, b) => a.d2 - b.d2);
  const yaws = shuffled(
    Array.from({ length: SCENE_YAW_CANDIDATES }, (_, k) => ((k + rng()) * 360) / SCENE_YAW_CANDIDATES),
    rng,
  ).map((d) => d * DEG);

  const deadline = Date.now() + SCENE_SEARCH_BUDGET_MS;
  const rejected = { spot: 0, observer: 0, corridor: 0, los: 0, view: 0 };
  let candidates = 0;
  for (const col of columns) {
    if (Date.now() > deadline) break;
    for (let dy = 1; dy >= -2; dy--) {
      const y = base.y + dy;
      const a = { x: base.x + col.dx + 0.5, y, z: base.z + col.dz + 0.5 };
      if (!isStandable(bot, a.x, a.y, a.z)) continue;
      if (!isSpotClear(bot, a)) {
        rejected.spot++;
        continue;
      }
      for (const yawA of yaws) {
        candidates++;
        const toB = keyDirection("forward", yawA);
        const b = { x: a.x + toB.x * dist, y, z: a.z + toB.z * dist };
        if (!isStandable(bot, b.x, b.y, b.z) || !isSpotClear(bot, b)) {
          rejected.observer++;
          continue;
        }
        // Level strip between the two bots (also the forward corridor).
        if (!isCorridorFree(bot, a, toB, dist)) {
          rejected.corridor++;
          continue;
        }
        if (KEY_OF[action] !== "jump" && KEY_OF[action] !== "sneak") {
          const dir = keyDirection(KEY_OF[action], yawA);
          if (!isCorridorFree(bot, a, dir, WALK_BLOCKS + CORRIDOR_MARGIN_BLOCKS)) {
            rejected.corridor++;
            continue;
          }
        }
        const eyeA = eyeOf(a);
        const eyeB = eyeOf(b);
        if (!bodyVisibleFrom(bot, eyeB, a) || !bodyVisibleFrom(bot, eyeA, b)) {
          rejected.los++;
          continue;
        }
        const yawB = anglesTo(eyeB, eyeA).yaw;
        if (!isViewGood(bot, b, yawB) || !isViewGood(bot, a, yawA)) {
          rejected.view++;
          continue;
        }
        return {
          actorPos: new Vec3(a.x, a.y, a.z),
          observerPos: new Vec3(b.x, b.y, b.z),
          yawA,
          yawB,
          candidates,
          rejected,
        };
      }
      // One level per column is enough: the first standable y is the surface.
      break;
    }
    await sleep(0);
  }
  console.log(`[${bot.username}] scene search: no layout among ${candidates} candidates, rejected by ${JSON.stringify(rejected)}`);
  return null;
}

/**
 * Move the actor to a new search spot: on odd attempts the nearest flat
 * biome (`locate biome`) of a random point within range of the original
 * position, on even attempts a random spot; the bot lands on the surface
 * within HOP_SPREAD_RADIUS of that point (same scheme as coObserveEval).
 */
async function hop(bot, rcon, origin, attempt, rng) {
  const range = Math.min(SCENE_HOP_RANGE * 2 ** (attempt - 1), SCENE_HOP_MAX_RANGE);
  const ang = rng() * 2 * Math.PI;
  const d = range * (0.5 + 0.5 * rng());
  let x = Math.floor(origin.x + d * Math.cos(ang));
  let z = Math.floor(origin.z + d * Math.sin(ang));
  let how = `random within ${range}`;
  if (attempt % 2 === 1) {
    const biome = FLAT_BIOMES[Math.floor((attempt - 1) / 2) % FLAT_BIOMES.length];
    try {
      const res = String(await rcon.send(`execute positioned ${x} 64 ${z} run locate biome minecraft:${biome}`));
      const m = /\[(-?\d+), (?:~|-?\d+), (-?\d+)\]/.exec(res);
      if (m) {
        x = parseInt(m[1], 10);
        z = parseInt(m[2], 10);
        how = `${biome} (${res.trim()})`;
      } else {
        how = `${biome} not found (${res.trim()}), random`;
      }
    } catch (err) {
      how = `locate failed: ${err.message}, random`;
    }
  }
  let res = "";
  for (let i = 0; i < 3; i++) {
    res = String(await rcon.send(`spreadplayers ${x} ${z} 0 ${HOP_SPREAD_RADIUS} false ${bot.username}`)).trim();
    if (/^Spread/.test(res)) break;
    const a = rng() * 2 * Math.PI;
    x = Math.floor(origin.x + range * Math.cos(a));
    z = Math.floor(origin.z + range * Math.sin(a));
    how += `; retry at ${x},${z}`;
  }
  console.log(`[${bot.username}] scene hop ${attempt}: ${how} -> ${res}`);
}

// ---------------------------------------------------------------------------
// The action
// ---------------------------------------------------------------------------

function posArray(p) {
  return [p.x, p.y, p.z];
}

// Run the one action with the camera untouched. Fills the timing and the
// poses into `ev`: start_time_ms is the key press, key_release_time_ms the
// release, end_time_ms the end of the visible motion (the bot came to rest /
// landed / stood up).
async function doAction(bot, ev) {
  const key = KEY_OF[ev.type];
  const start = bot.entity.position.clone();
  ev.actor_pos_before = posArray(start);
  ev.yaw_before = bot.entity.yaw;
  ev.pitch_before = bot.entity.pitch;
  let maxY = start.y;
  ev.start_time_ms = Date.now();
  if (key === "jump") {
    bot.setControlState("jump", true);
    await bot.waitForTicks(1);
    bot.setControlState("jump", false);
    ev.key_release_time_ms = Date.now();
    let airborne = false;
    let ticks = 1;
    while (ticks < JUMP_MAX_TICKS) {
      await bot.waitForTicks(1);
      ticks++;
      maxY = Math.max(maxY, bot.entity.position.y);
      if (!bot.entity.onGround) airborne = true;
      else if (airborne) break;
    }
    ev.airborne = airborne;
    ev.motion_ticks = ticks;
  } else if (key === "sneak") {
    bot.setControlState("sneak", true);
    await bot.waitForTicks(SNEAK_TICKS);
    bot.setControlState("sneak", false);
    ev.key_release_time_ms = Date.now();
    await bot.waitForTicks(1);
    ev.motion_ticks = SNEAK_TICKS + 1;
  } else {
    bot.setControlState(key, true);
    await bot.waitForTicks(WALK_TICKS);
    bot.setControlState(key, false);
    ev.key_release_time_ms = Date.now();
    // The bot slides for a few ticks after the key is released.
    let ticks = WALK_TICKS;
    let last = bot.entity.position.clone();
    for (let i = 0; i < SLIDE_MAX_TICKS; i++) {
      await bot.waitForTicks(1);
      ticks++;
      const now = bot.entity.position.clone();
      if (now.distanceTo(last) < SLIDE_STOP_EPS) break;
      last = now;
    }
    ev.motion_ticks = ticks;
  }
  ev.end_time_ms = Date.now();
  const end = bot.entity.position.clone();
  ev.actor_pos_after = posArray(end);
  ev.displacement = [end.x - start.x, end.y - start.y, end.z - start.z];
  ev.displacement_blocks = Math.hypot(end.x - start.x, end.z - start.z);
  ev.max_y_rise = maxY - start.y;
  ev.yaw_after = bot.entity.yaw;
  ev.pitch_after = bot.entity.pitch;
  // Displacement in the actor's frame: along its look direction and to its
  // left.
  const fwd = keyDirection("forward", ev.yaw_before);
  const left = keyDirection("left", ev.yaw_before);
  ev.displacement_forward = ev.displacement[0] * fwd.x + ev.displacement[2] * fwd.z;
  ev.displacement_left = ev.displacement[0] * left.x + ev.displacement[2] * left.z;
}

// ---------------------------------------------------------------------------
// Phase
// ---------------------------------------------------------------------------

function getOnPhaseFn(bot, rcon, sharedBotRng, coordinator, episodeNum, episodeInstance, args) {
  return async () => {
    coordinator.sendToOtherBot("actorMotionPhase", bot.entity.position.clone(), episodeNum, "actorMotionPhase beginning");
    const layout = episodeInstance._layout;
    const stopNow = () => {
      coordinator.onceEvent(
        "stopPhase",
        episodeNum,
        episodeInstance.getOnStopPhaseFn(bot, rcon, sharedBotRng, coordinator, args.other_bot_name, episodeNum, args),
      );
      coordinator.sendToOtherBot("stopPhase", bot.entity.position.clone(), episodeNum, "actorMotionPhase end");
    };

    if (!layout) {
      console.log(`[${bot.username}] no layout (scene search failed), stopping`);
      stopNow();
      return;
    }
    const ev = layout.event;

    if (!episodeInstance._isActor) {
      // Observer: aim at the actor's body centre, hold still, wait for the
      // actor's report.
      coordinator.onceEvent("actorMotionDone", episodeNum, (data) => {
        const meta = episodeInstance._episodeMetadata;
        meta.events = [data.event];
        meta.event = data.event;
        meta.window_start_time_ms = data.window_start_time_ms;
        meta.window_end_time_ms = data.window_end_time_ms;
        console.log(`[${bot.username}] received the ${data.event.type} event from the actor`);
        stopNow();
      });
      stopAll(bot);
      await lookExact(bot, lookState(bot), anglesTo(eyeOf(bot.entity.position), bodyCentreOf(layout.actorPos)));
      return;
    }

    // Actor: aim once, hold, act, hold.
    const look = lookState(bot);
    const eyeA = eyeOf(layout.actorPos);
    const eyeB = eyeOf(layout.observerPos);
    let windowStart = null;
    let windowEnd = null;
    try {
      stopAll(bot);
      await lookExact(bot, look, anglesTo(eyeA, bodyCentreOf(layout.observerPos)));
      await bot.waitForTicks(2);
      windowStart = Date.now();
      await bot.waitForTicks(PRE_HOLD_TICKS);

      try {
        await doAction(bot, ev);
        ev.error = null;
      } catch (err) {
        ev.error = err.message;
        ev.end_time_ms = ev.end_time_ms ?? Date.now();
        console.log(`[${bot.username}] ${ev.type} failed: ${err.message}`);
      } finally {
        stopAll(bot);
      }
      ev.los_clear = {
        before: bodyVisibleFrom(bot, eyeB, layout.actorPos),
        after: bodyVisibleFrom(bot, eyeB, bot.entity.position),
      };
      if (!ev.error) {
        console.log(
          `[${bot.username}] ${ev.type}: ${ev.motion_ticks} ticks, moved ${ev.displacement_blocks.toFixed(2)} blocks ` +
            `(forward ${ev.displacement_forward.toFixed(2)}, left ${ev.displacement_left.toFixed(2)}, ` +
            `rise ${ev.max_y_rise.toFixed(2)}), yaw drift ${((ev.yaw_after - ev.yaw_before) / DEG).toFixed(3)} deg`,
        );
      }
      const usedTicks = Math.round((Date.now() - windowStart) / 50);
      await bot.waitForTicks(Math.max(POST_MIN_TICKS, WINDOW_TARGET_TICKS - usedTicks));
      windowEnd = Date.now();
    } finally {
      stopAll(bot);
    }

    const meta = episodeInstance._episodeMetadata;
    meta.events = [ev];
    meta.event = ev;
    meta.window_start_time_ms = windowStart;
    meta.window_end_time_ms = windowEnd;
    coordinator.sendToOtherBot(
      "actorMotionDone",
      { event: ev, window_start_time_ms: windowStart, window_end_time_ms: windowEnd },
      episodeNum,
      "actorMotionPhase actor done",
    );
    stopNow();
  };
}

/**
 * Actor-motion eval episode (state propagation of body motion). Bot A
 * (actor) performs one atomic action, forward / back / left / right walk,
 * one jump or one sneak, without turning, while bot B (observer) stands
 * still OBSERVER_DIST_RANGE blocks away on flat open ground, facing A with A
 * at the centre of its frame; A faces B, so A's "forward" is towards B.
 * Both hold still before (PRE_HOLD_TICKS) and after the action, the window
 * being padded to WINDOW_TARGET_TICKS.
 *
 * episode_info carries `action`, `events` = [event] (the shared postprocess
 * fills frame_start / frame_end): type, key, timing (start_time_ms = key
 * press, key_release_time_ms, end_time_ms = motion end), the actor's pose
 * before / after, displacement (world and actor frame), max_y_rise, line of
 * sight from B; the window (window_start_time_ms / window_end_time_ms ->
 * window_start_frame / window_end_frame), both poses and the distance.
 * @extends BaseEpisode
 */
class ActorMotionEvalEpisode extends BaseEpisode {
  static WORKS_IN_NON_FLAT_WORLD = true;
  static INIT_MIN_BOTS_DISTANCE = 4;
  static INIT_MAX_BOTS_DISTANCE = 8;

  async setupEpisode(bot, rcon, sharedBotRng, coordinator, episodeNum, args, botPosition, otherBotPosition) {
    const seed = `actorMotionEval_${ACTOR_MOTION_BASE_SEED}_${args.instance_id ?? 0}_${episodeNum}`;
    const isActor = decidePrimaryBot(bot, sharedBotRng, args);
    this._isActor = isActor;
    this._layout = null;
    const actorName = isActor ? bot.username : args.other_bot_name;
    const observerName = isActor ? args.other_bot_name : bot.username;
    this._episodeMetadata = {
      actor_motion_seed: seed,
      actor: actorName,
      observer: observerName,
      role: isActor ? "actor" : "observer",
      pre_hold_ticks: PRE_HOLD_TICKS,
      walk_ticks: WALK_TICKS,
      sneak_ticks: SNEAK_TICKS,
      post_min_ticks: POST_MIN_TICKS,
      window_target_ticks: WINDOW_TARGET_TICKS,
      scene_check_passed: false,
      events: [],
    };
    this._evalMetadata = { bots_chosen: [bot.username, args.other_bot_name].sort(), actor: actorName };

    await quietNotificationsOnce(bot, rcon);

    let layoutPromise = null;
    if (!isActor) {
      layoutPromise = new Promise((resolve) => {
        coordinator.onceEvent("actorMotionLayout", episodeNum, (data) => resolve(data));
      });
    }
    await rendezvous(coordinator, "actorMotionReady", episodeNum);

    let layout = null;
    if (isActor) {
      layout = await this.actorSetup(bot, rcon, episodeNum, args, botPosition, seed);
      coordinator.sendToOtherBot("actorMotionLayout", { layout }, episodeNum, "actorMotionLayout send");
    } else {
      layout = (await layoutPromise).layout;
      if (layout) await sleep(OBSERVER_CHUNK_SETTLE_MS);
    }
    if (!layout) {
      return { botPositionNew: botPosition, otherBotPositionNew: otherBotPosition };
    }
    this._layout = layout;
    Object.assign(this._episodeMetadata, {
      scene_check_passed: true,
      action: layout.event.type,
      action_key: layout.event.key,
      actor_pos: [layout.actorPos.x, layout.actorPos.y, layout.actorPos.z],
      observer_pos: [layout.observerPos.x, layout.observerPos.y, layout.observerPos.z],
      actor_yaw: layout.yawA,
      observer_yaw: layout.yawB,
      observer_dist: layout.dist,
      scene_search_attempts: layout.attempts,
      events: [layout.event],
      event: layout.event,
      suggested_window: "window_start_frame .. window_end_frame (20 static frames, the action, >= 40 static frames)",
    });
    Object.assign(this._evalMetadata, { action: layout.event.type, observer_dist: layout.dist });

    // The caller aims the camera at otherBotPositionNew with a feet-based
    // look; shift the aim point down by the eye height so the resulting
    // yaw/pitch are the eye-based ones (both bots aim at the other's body
    // centre).
    const aim = bodyCentreOf(isActor ? layout.observerPos : layout.actorPos);
    const myPos = isActor ? layout.actorPos : layout.observerPos;
    return {
      botPositionNew: new Vec3(myPos.x, myPos.y, myPos.z),
      otherBotPositionNew: new Vec3(aim.x, aim.y - EYE_HEIGHT, aim.z),
    };
  }

  async actorSetup(bot, rcon, episodeNum, args, botPosition, seed) {
    const rng = seedrandom(seed);
    const action = ACTIONS[((args.instance_id ?? 0) + episodeNum) % ACTIONS.length];
    const dist = OBSERVER_DIST_RANGE[0] + rng() * (OBSERVER_DIST_RANGE[1] - OBSERVER_DIST_RANGE[0]);
    console.log(`[${bot.username}] Episode ${episodeNum}: actor, action ${action}, observer at ${dist.toFixed(2)} blocks`);

    let scene = null;
    let attempts = 0;
    while (scene === null && attempts < SCENE_MAX_ATTEMPTS) {
      if (attempts > 0) {
        await hop(bot, rcon, botPosition, attempts, rng);
        await sleep(SCENE_HOP_SETTLE_MS);
      }
      attempts++;
      scene = await pickScene(bot, rng, action, dist);
    }
    this._episodeMetadata.scene_search_attempts = attempts;
    if (scene === null) return null;
    const { actorPos, observerPos } = scene;
    console.log(
      `[${bot.username}] scene: actor ${actorPos} yaw ${(scene.yawA / DEG).toFixed(1)} deg, observer ${observerPos} ` +
        `(${dist.toFixed(2)} blocks) after ${scene.candidates} candidates, ${attempts} attempt(s)`,
    );

    for (const [name, pos] of [[bot.username, actorPos], [args.other_bot_name, observerPos]]) {
      const res = await rcon.send(`tp ${name} ${pos.x} ${pos.y} ${pos.z}`);
      console.log(`[${bot.username}] tp ${name} -> ${String(res).trim()}`);
    }
    await sleep(1500);
    if (bot.entity.position.distanceTo(actorPos) > 0.3) {
      console.log(`[${bot.username}] actor not at scene start (${bot.entity.position}), giving up`);
      return null;
    }

    // Empty hands in both views, no mobs, no weather.
    const cx = Math.floor(actorPos.x);
    const cz = Math.floor(actorPos.z);
    for (const cmd of [
      `clear ${bot.username}`,
      `clear ${args.other_bot_name}`,
      "gamerule doMobSpawning false",
      "weather clear 1000000",
      `execute positioned ${cx} ${actorPos.y} ${cz} run kill @e[type=!minecraft:player,distance=..${MOB_CLEAR_RADIUS}]`,
    ]) {
      const res = await rcon.send(cmd);
      console.log(`[${bot.username}] ${cmd} -> ${String(res).trim()}`);
    }

    return {
      actorPos,
      observerPos,
      yawA: scene.yawA,
      yawB: scene.yawB,
      dist,
      attempts,
      event: { idx: 0, type: action, key: KEY_OF[action], actor: bot.username },
    };
  }

  async entryPoint(bot, rcon, sharedBotRng, coordinator, iterationID, episodeNum, args) {
    coordinator.onceEvent("actorMotionPhase", episodeNum, getOnPhaseFn(bot, rcon, sharedBotRng, coordinator, episodeNum, this, args));
    coordinator.sendToOtherBot("actorMotionPhase", bot.entity.position.clone(), episodeNum, "entryPoint end");
  }

  async tearDownEpisode(bot, rcon, sharedBotRng, coordinator, episodeNum, args) {
    if (!this._isActor) return;
    try {
      await rcon.send("gamerule doMobSpawning true");
    } catch (err) {
      console.log(`[${bot.username}] teardown failed: ${err.message}`);
    }
  }
}

module.exports = { ActorMotionEvalEpisode };

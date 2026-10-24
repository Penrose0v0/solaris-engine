const seedrandom = require("seedrandom");
const { Vec3 } = require("vec3");

const { stopAll } = require("../../primitives/movement");
const { decidePrimaryBot } = require("../../utils/coordination");
const { sleep } = require("../../utils/helpers");
const { BaseEpisode } = require("../base-episode");
const { sceneUtils } = require("./traj-segments-eval-episode");
const { coObserveUtils } = require("./co-observe-eval-episode");

const { isStandable, EYE_HEIGHT } = sceneUtils;
const {
  surfaceMap,
  rateArena,
  terraform,
  repairArena,
  hop,
  quietNotificationsOnce,
  rendezvous,
  anglesTo,
  eyeOf,
  wrapRad,
  BLOCK_TYPES,
  TERRAIN_STRICT,
  TERRAIN_RELAXED,
  RELAXED_FROM_ATTEMPT,
  SCENE_MAX_ATTEMPTS,
  SCENE_HOP_SETTLE_MS,
  SCENE_SEARCH_RADIUS,
  MAX_UNLOADED_FRACTION,
  HEIGHT_MAP_RETRIES,
  HEIGHT_MAP_RETRY_MS,
  MOB_CLEAR_RADIUS,
  DEG,
} = coObserveUtils;

// Fixed base seed; the per-episode seed is derived from (base seed, instance
// id, episode number) so the plan is reproducible.
const SHARED_ARENA_BASE_SEED = 20261024;

// Enclosure: square walls one block thick around an INNER_SIZE x INNER_SIZE
// floor, WALL_HEIGHT_RANGE tall, no roof. Every wall and floor block gets a
// random concrete colour (never equal to its left / lower neighbour), so no
// stretch of wall repeats. The levelled square reaches ARENA_MARGIN beyond
// the walls and is cleared up to the wall top plus one.
const INNER_SIZE = 15;
const WALL_HEIGHT_RANGE = [4, 5];
const ARENA_MARGIN = 1;
const INNER_HALF = (INNER_SIZE - 1) / 2; // 7: floor columns |dx|,|dz| <= 7
const WALL_OFFSET = INNER_HALF + 1; // 8: wall columns at |dx| == 8 or |dz| == 8
const ARENA_HALF = WALL_OFFSET + ARENA_MARGIN; // 9

// Bots: random cells at least WALL_CLEARANCE blocks from the walls, BOT_DISTANCE
// apart. The first bot looks in a random direction; the second looks at the
// same wall region (A's wall hit point, shifted along the wall by up to
// OVERLAP_SHIFT_BLOCKS, then turned by up to OVERLAP_YAW_JITTER_DEG).
const WALL_CLEARANCE = 3;
const BOT_DISTANCE = [4, 8];
const OVERLAP_SHIFT_BLOCKS = 2.0;
const OVERLAP_YAW_JITTER_DEG = 15;
const INITIAL_PITCH_RANGE_DEG = [-5, 5];

// Motion (physics ticks, ~1 frame each): LEAD_TICKS static, MOTION_TICKS of
// yaw sweeping (the eval window), TAIL_TICKS static. The yaw follows a
// triangle wave around the initial yaw with peak-to-peak amplitude
// SWEEP_RANGE_DEG at OMEGA_RANGE_DEG_PER_SEC (multiples of 3 deg/s so a
// tick step is a multiple of mineflayer's 0.15 deg look quantum); the pitch
// adds a small sine (PITCH_AMP_RANGE_DEG, PITCH_PERIOD_RANGE_S). No
// translation.
const LEAD_TICKS = 10;
const MOTION_TICKS = 160;
const TAIL_TICKS = 25;
const SWEEP_RANGE_DEG = [90, 270];
const OMEGA_RANGE_DEG_PER_SEC = [30, 90];
const OMEGA_QUANTUM_DEG_PER_SEC = 3;
const PITCH_AMP_RANGE_DEG = [0, 10];
const PITCH_PERIOD_RANGE_S = [3, 6];
const PITCH_LIMIT_DEG = 60;

const SCENE_SEARCH_BUDGET_MS = 6000;
const SECONDARY_CHUNK_SETTLE_MS = 2500;
const BUILD_VERIFY_TIMEOUT_MS = 8000;
const BUILD_SERVER_SAMPLE = 40;

// ---------------------------------------------------------------------------
// Plan (mineflayer convention: yaw 0 = north = -z, yaw grows counter-
// clockwise, pitch positive = up)
// ---------------------------------------------------------------------------

function uniform(rng, lo, hi) {
  return lo + rng() * (hi - lo);
}

function viewDir(yaw) {
  return { x: -Math.sin(yaw), z: -Math.cos(yaw) };
}

function yawTo(from, to) {
  return Math.atan2(-(to.x - from.x), -(to.z - from.z));
}

/**
 * Point where a horizontal ray from `pos` along `yaw` meets the inner wall
 * faces (at centre +- (WALL_OFFSET - 0.5) on both axes). Returns
 * {x, z, wall} with wall in east/west/south/north (+x/-x/+z/-z).
 */
function wallHit(centre, pos, yaw) {
  const d = viewDir(yaw);
  const lim = WALL_OFFSET - 0.5;
  let best = null;
  const planes = [
    { axis: "x", value: centre.x + lim, wall: "east" },
    { axis: "x", value: centre.x - lim, wall: "west" },
    { axis: "z", value: centre.z + lim, wall: "south" },
    { axis: "z", value: centre.z - lim, wall: "north" },
  ];
  for (const p of planes) {
    const dirc = p.axis === "x" ? d.x : d.z;
    const posc = p.axis === "x" ? pos.x : pos.z;
    if (Math.abs(dirc) < 1e-9) continue;
    const t = (p.value - posc) / dirc;
    if (t <= 0) continue;
    const hx = pos.x + d.x * t;
    const hz = pos.z + d.z * t;
    const other = p.axis === "x" ? hz - centre.z : hx - centre.x;
    if (Math.abs(other) > lim + 1e-6) continue;
    if (best === null || t < best.t) best = { x: hx, z: hz, wall: p.wall, t };
  }
  return best;
}

function wallIndex(wall) {
  return { east: 0, south: 1, west: 2, north: 3 }[wall];
}

function wallsAdjacent(a, b) {
  return a === b || (wallIndex(a) - wallIndex(b) + 4) % 2 === 1;
}

/**
 * Enclosure blocks relative to the centre column: floor at dy -1 inside,
 * walls at |dx| == WALL_OFFSET or |dz| == WALL_OFFSET from dy 0 to
 * height-1. Colours random per block, never equal to the block to the
 * "left" (previous in build order along the wall) or below.
 */
function buildEnclosureShape(rng) {
  const height = WALL_HEIGHT_RANGE[0] + Math.floor(rng() * (WALL_HEIGHT_RANGE[1] - WALL_HEIGHT_RANGE[0] + 1));
  const blocks = [];
  const colour = new Map();
  const pick = (...avoid) => {
    let c;
    do {
      c = BLOCK_TYPES[Math.floor(rng() * BLOCK_TYPES.length)];
    } while (avoid.includes(c));
    return c;
  };
  const key = (dx, dy, dz) => `${dx},${dy},${dz}`;
  // Floor.
  for (let dx = -INNER_HALF; dx <= INNER_HALF; dx++) {
    for (let dz = -INNER_HALF; dz <= INNER_HALF; dz++) {
      const c = pick(colour.get(key(dx - 1, -1, dz)), colour.get(key(dx, -1, dz - 1)));
      colour.set(key(dx, -1, dz), c);
      blocks.push({ dx, dy: -1, dz, block: c, part: "floor" });
    }
  }
  // Walls, column by column around the ring.
  const ring = [];
  for (let dx = -WALL_OFFSET; dx <= WALL_OFFSET; dx++) ring.push([dx, -WALL_OFFSET]);
  for (let dz = -WALL_OFFSET + 1; dz <= WALL_OFFSET; dz++) ring.push([WALL_OFFSET, dz]);
  for (let dx = WALL_OFFSET - 1; dx >= -WALL_OFFSET; dx--) ring.push([dx, WALL_OFFSET]);
  for (let dz = WALL_OFFSET - 1; dz >= -WALL_OFFSET + 1; dz--) ring.push([-WALL_OFFSET, dz]);
  let prev = null;
  for (const [dx, dz] of ring) {
    for (let dy = 0; dy < height; dy++) {
      const left = prev ? colour.get(key(prev[0], dy, prev[1])) : undefined;
      const c = pick(left, colour.get(key(dx, dy - 1, dz)));
      colour.set(key(dx, dy, dz), c);
      blocks.push({ dx, dy, dz, block: c, part: "wall" });
    }
    prev = [dx, dz];
  }
  return { height, blocks };
}

/**
 * Per-episode plan: enclosure shape, bot cells, initial looks (B overlaps
 * A's wall region) and sweep parameters.
 */
function buildPlan(rng) {
  const shape = buildEnclosureShape(rng);
  // Cell indices |dx|,|dz| <= 4: the inner wall faces sit at +-7.5 from the
  // centre, a bot at cell 4 is 3.5 blocks from the nearest wall.
  const lim = Math.floor(WALL_OFFSET - 0.5 - WALL_CLEARANCE);
  const cell = () => ({
    dx: Math.floor(uniform(rng, -lim, lim + 1)),
    dz: Math.floor(uniform(rng, -lim, lim + 1)),
  });
  let a = cell();
  let b = cell();
  for (let tries = 0; tries < 200; tries++) {
    const d = Math.hypot(a.dx - b.dx, a.dz - b.dz);
    if (d >= BOT_DISTANCE[0] && d <= BOT_DISTANCE[1]) break;
    b = cell();
    if (tries % 20 === 19) a = cell();
  }
  const sweep = (name) => {
    const kMin = Math.ceil(OMEGA_RANGE_DEG_PER_SEC[0] / OMEGA_QUANTUM_DEG_PER_SEC);
    const kMax = Math.floor(OMEGA_RANGE_DEG_PER_SEC[1] / OMEGA_QUANTUM_DEG_PER_SEC);
    return {
      name,
      sweep_deg: uniform(rng, SWEEP_RANGE_DEG[0], SWEEP_RANGE_DEG[1]),
      omega_deg_per_sec: OMEGA_QUANTUM_DEG_PER_SEC * Math.floor(uniform(rng, kMin, kMax + 1)),
      direction: rng() < 0.5 ? 1 : -1,
      pitch_amp_deg: uniform(rng, PITCH_AMP_RANGE_DEG[0], PITCH_AMP_RANGE_DEG[1]),
      pitch_period_s: uniform(rng, PITCH_PERIOD_RANGE_S[0], PITCH_PERIOD_RANGE_S[1]),
      pitch_phase: rng() * 2 * Math.PI,
    };
  };
  return {
    shape,
    cells: { a, b },
    yaw0A: uniform(rng, -Math.PI, Math.PI),
    pitch0: [uniform(rng, INITIAL_PITCH_RANGE_DEG[0], INITIAL_PITCH_RANGE_DEG[1]) * DEG,
      uniform(rng, INITIAL_PITCH_RANGE_DEG[0], INITIAL_PITCH_RANGE_DEG[1]) * DEG],
    overlap_shift: uniform(rng, -OVERLAP_SHIFT_BLOCKS, OVERLAP_SHIFT_BLOCKS),
    overlap_jitter: uniform(rng, -OVERLAP_YAW_JITTER_DEG, OVERLAP_YAW_JITTER_DEG) * DEG,
    sweeps: [sweep("a"), sweep("b")],
  };
}

// Yaw offset (rad) of the triangle sweep after `travelled` degrees of
// rotation: 0 -> +S/2 -> -S/2 -> 0 ... (times `direction`).
function triangleOffsetDeg(travelled, sweepDeg) {
  const period = 2 * sweepDeg;
  const p = ((travelled % period) + period) % period;
  if (p <= sweepDeg / 2) return p;
  if (p <= 1.5 * sweepDeg) return sweepDeg - p;
  return p - 2 * sweepDeg;
}

function lookAtTick(me, t) {
  const travelled = (me.omega_deg_per_sec / 20) * t;
  const yaw = me.yaw0 + me.direction * triangleOffsetDeg(travelled, me.sweep_deg) * DEG;
  const pitchDeg =
    me.pitch0 / DEG + me.pitch_amp_deg * Math.sin((2 * Math.PI * t) / (20 * me.pitch_period_s) + me.pitch_phase);
  const pitch = Math.max(-PITCH_LIMIT_DEG, Math.min(PITCH_LIMIT_DEG, pitchDeg)) * DEG;
  return { yaw, pitch };
}

// ---------------------------------------------------------------------------
// Scene selection: a nearly level square of half-size ARENA_HALF (see
// coObserveUtils.rateArena); no view constraints, the walls are the scene.
// ---------------------------------------------------------------------------

async function pickArena(bot, limits) {
  const base = bot.entity.position.floored();
  let surface = null;
  for (let i = 0; i <= HEIGHT_MAP_RETRIES; i++) {
    surface = surfaceMap(bot, base, SCENE_SEARCH_RADIUS + ARENA_HALF);
    if (surface.unloaded / surface.total <= MAX_UNLOADED_FRACTION) break;
    console.log(`[${bot.username}] surface map: ${surface.unloaded}/${surface.total} columns unloaded, waiting for chunks`);
    await sleep(HEIGHT_MAP_RETRY_MS);
  }
  const rejected = { centre: 0, unloaded: 0, cliff: 0, water: 0, odd: 0, rough: 0, raised: 0 };
  const rated = [];
  const deadline = Date.now() + SCENE_SEARCH_BUDGET_MS;
  for (let dx = -SCENE_SEARCH_RADIUS; dx <= SCENE_SEARCH_RADIUS; dx++) {
    for (let dz = -SCENE_SEARCH_RADIUS; dz <= SCENE_SEARCH_RADIUS; dz++) {
      if (Date.now() > deadline) break;
      const e = surface.map.get(`${dx},${dz}`);
      if (!e || e.h === null || e.liquid || e.odd) {
        rejected.centre++;
        continue;
      }
      const r = rateArena(surface, dx, dz, e.h, ARENA_HALF, limits);
      if (r.reason) {
        rejected[r.reason]++;
        continue;
      }
      rated.push({ dx, dz, g: e.h, stats: r.stats, d2: dx * dx + dz * dz });
    }
    await sleep(0);
  }
  if (rated.length === 0) {
    console.log(
      `[${bot.username}] arena search: nothing level enough, rejected by ${JSON.stringify(rejected)}, unloaded ${surface.unloaded}/${surface.total}`,
    );
    return null;
  }
  rated.sort(
    (a, b) =>
      b.stats.level_fraction - a.stats.level_fraction ||
      a.stats.raised_fraction - b.stats.raised_fraction ||
      a.d2 - b.d2,
  );
  const best = rated[0];
  const cx = base.x + best.dx;
  const cz = base.z + best.dz;
  const y = best.g + 1;
  return { centre: { x: cx + 0.5, y, z: cz + 0.5 }, arena: { cx, cz, y, half: ARENA_HALF }, terrain: best.stats, rated: rated.length };
}

// ---------------------------------------------------------------------------
// RCON: enclosure
// ---------------------------------------------------------------------------

async function rconBlockIs(rcon, b) {
  const res = await rcon.send(`execute if block ${b.x} ${b.y} ${b.z} minecraft:${b.block}`);
  return /passed/i.test(String(res));
}

async function buildEnclosure(bot, rcon, blocks, rng) {
  let failed = 0;
  for (const b of blocks) {
    const res = String(await rcon.send(`setblock ${b.x} ${b.y} ${b.z} minecraft:${b.block}`));
    if (!/Changed/i.test(res)) failed++;
  }
  // Server: a random sample; client: every block.
  let serverBad = 0;
  for (let i = 0; i < BUILD_SERVER_SAMPLE; i++) {
    const b = blocks[Math.floor(rng() * blocks.length)];
    if (!(await rconBlockIs(rcon, b))) serverBad++;
  }
  const deadline = Date.now() + BUILD_VERIFY_TIMEOUT_MS;
  let missing = blocks.length;
  while (missing > 0 && Date.now() < deadline) {
    missing = blocks.filter((b) => {
      const c = bot.blockAt(new Vec3(b.x, b.y, b.z));
      return !(c && c.name === b.block);
    }).length;
    if (missing > 0) await sleep(200);
  }
  console.log(
    `[${bot.username}] enclosure ${blocks.length} blocks: setblock not-changed ${failed}, server sample bad ${serverBad}/${BUILD_SERVER_SAMPLE}, client missing ${missing}`,
  );
  return { ok: serverBad === 0 && missing === 0, setblock_not_changed: failed, server_sample_bad: serverBad, client_missing: missing };
}

// ---------------------------------------------------------------------------
// Phase
// ---------------------------------------------------------------------------

function getOnSharedArenaPhaseFn(bot, rcon, sharedBotRng, coordinator, episodeNum, episodeInstance, args) {
  return async () => {
    coordinator.sendToOtherBot("sharedArenaPhase", bot.entity.position.clone(), episodeNum, "sharedArenaPhase beginning");
    const layout = episodeInstance._layout;
    const stopNow = () => {
      coordinator.onceEvent(
        "stopPhase",
        episodeNum,
        episodeInstance.getOnStopPhaseFn(bot, rcon, sharedBotRng, coordinator, args.other_bot_name, episodeNum, args),
      );
      coordinator.sendToOtherBot("stopPhase", bot.entity.position.clone(), episodeNum, "sharedArenaPhase end");
    };
    if (!layout) {
      console.log(`[${bot.username}] no layout (arena search failed), stopping`);
      stopNow();
      return;
    }
    const me = layout.bots[bot.username];
    const meta = episodeInstance._episodeMetadata;
    const trace = [];
    const start = bot.entity.position.clone();
    try {
      await bot.look(me.yaw0, me.pitch0, true);
      await bot.waitForTicks(LEAD_TICKS);
      await rendezvous(coordinator, "sharedArenaGo", episodeNum);
      meta.window_start_time_ms = Date.now();
      for (let t = 0; t < MOTION_TICKS; t++) {
        const { yaw, pitch } = lookAtTick(me, t);
        await bot.look(yaw, pitch, true);
        trace.push({ t, time_ms: Date.now(), yaw: +yaw.toFixed(4), pitch: +pitch.toFixed(4) });
        await bot.waitForTicks(1);
      }
      meta.window_end_time_ms = Date.now();
      await bot.waitForTicks(TAIL_TICKS);
    } finally {
      stopAll(bot);
    }
    const end = bot.entity.position;
    meta.look_trace = trace;
    meta.drift_blocks = Math.hypot(end.x - start.x, end.y - start.y, end.z - start.z);
    const yaws = trace.map((p) => p.yaw);
    meta.sweep_stats = {
      yaw_min_deg: Math.min(...yaws) / DEG,
      yaw_max_deg: Math.max(...yaws) / DEG,
      yaw_travel_deg: trace.slice(1).reduce((s, p, i) => s + Math.abs(p.yaw - trace[i].yaw), 0) / DEG,
      pitch_min_deg: Math.min(...trace.map((p) => p.pitch)) / DEG,
      pitch_max_deg: Math.max(...trace.map((p) => p.pitch)) / DEG,
    };
    console.log(`[${bot.username}] sweep done: ${JSON.stringify(meta.sweep_stats)} drift ${meta.drift_blocks.toFixed(3)}`);
    stopNow();
  };
}

/**
 * Shared-arena eval episode. A square enclosure (INNER_SIZE x INNER_SIZE
 * inside, walls WALL_HEIGHT_RANGE tall, no roof) with random per-block
 * concrete colours on the walls and the floor is built on a levelled spot
 * before the recording; both bots stand inside, BOT_DISTANCE apart and at
 * least WALL_CLEARANCE from the walls, look at the same wall region at the
 * start and then sweep their yaw back and forth (plus a small pitch sine)
 * without moving. The world does not change while recording.
 *
 * episode_info (both bots) carries the enclosure (centre, wall height, every
 * block), both bots' positions and look plans, the motion window
 * (window_start_time_ms / window_end_time_ms -> window_start_frame /
 * window_end_frame in postprocess, the suggested window start) and the
 * planned start-view overlap; each bot adds its own per-tick look trace.
 * @extends BaseEpisode
 */
class SharedArenaEvalEpisode extends BaseEpisode {
  static WORKS_IN_NON_FLAT_WORLD = true;
  static INIT_MIN_BOTS_DISTANCE = 4;
  static INIT_MAX_BOTS_DISTANCE = 8;

  async setupEpisode(bot, rcon, sharedBotRng, coordinator, episodeNum, args, botPosition, otherBotPosition) {
    const seed = `sharedArenaEval_${SHARED_ARENA_BASE_SEED}_${args.instance_id ?? 0}_${episodeNum}`;
    const isPrimary = decidePrimaryBot(bot, sharedBotRng, args);
    this._isPrimary = isPrimary;
    this._layout = null;
    const primaryName = isPrimary ? bot.username : args.other_bot_name;
    this._episodeMetadata = {
      shared_arena_seed: seed,
      primary: primaryName,
      role: isPrimary ? "primary" : "secondary",
      lead_ticks: LEAD_TICKS,
      motion_ticks: MOTION_TICKS,
      tail_ticks: TAIL_TICKS,
      scene_check_passed: false,
    };
    this._evalMetadata = { bots_chosen: [bot.username, args.other_bot_name].sort(), primary: primaryName };

    await quietNotificationsOnce(bot, rcon);

    let layoutPromise = null;
    if (!isPrimary) {
      layoutPromise = new Promise((resolve) => {
        coordinator.onceEvent("sharedArenaLayout", episodeNum, (data) => resolve(data));
      });
    }
    await rendezvous(coordinator, "sharedArenaReady", episodeNum);

    let layout = null;
    if (isPrimary) {
      layout = await this.primarySetup(bot, rcon, episodeNum, args, botPosition, seed);
      coordinator.sendToOtherBot("sharedArenaLayout", { layout }, episodeNum, "sharedArenaLayout send");
    } else {
      layout = (await layoutPromise).layout;
      if (layout) await sleep(SECONDARY_CHUNK_SETTLE_MS);
    }
    if (!layout) {
      return { botPositionNew: botPosition, otherBotPositionNew: otherBotPosition };
    }
    this._layout = layout;
    const me = layout.bots[bot.username];
    Object.assign(this._episodeMetadata, {
      scene_check_passed: true,
      centre: [layout.centre.x, layout.centre.y, layout.centre.z],
      enclosure: layout.enclosure,
      bots: layout.bots,
      start_overlap: layout.start_overlap,
      pos: me.pos,
      look_plan: { yaw0: me.yaw0, pitch0: me.pitch0, sweep_deg: me.sweep_deg, omega_deg_per_sec: me.omega_deg_per_sec,
        direction: me.direction, pitch_amp_deg: me.pitch_amp_deg, pitch_period_s: me.pitch_period_s, pitch_phase: me.pitch_phase },
      suggested_window: "window_start_frame .. window_end_frame (the 160-tick sweep); frames before it are the static lead",
      arena: layout.arena,
      terraform: layout.terraform,
      scene_search_attempts: layout.attempts,
      scene_hops: layout.hops,
    });
    Object.assign(this._evalMetadata, { sweep_deg: me.sweep_deg, omega_deg_per_sec: me.omega_deg_per_sec });

    // Initial look: the caller aims the camera at otherBotPositionNew with a
    // feet-based look, so hand it a point along the planned look direction
    // shifted down by the eye height.
    const p = me.pos;
    const eye = eyeOf({ x: p[0], y: p[1], z: p[2] });
    const d = 6;
    const aim = {
      x: eye.x - Math.sin(me.yaw0) * Math.cos(me.pitch0) * d,
      y: eye.y + Math.sin(me.pitch0) * d,
      z: eye.z - Math.cos(me.yaw0) * Math.cos(me.pitch0) * d,
    };
    return {
      botPositionNew: new Vec3(p[0], p[1], p[2]),
      otherBotPositionNew: new Vec3(aim.x, aim.y - EYE_HEIGHT, aim.z),
    };
  }

  async primarySetup(bot, rcon, episodeNum, args, botPosition, seed) {
    const rng = seedrandom(seed);
    const plan = buildPlan(rng);
    console.log(
      `[${bot.username}] Episode ${episodeNum}: primary, wall height ${plan.shape.height}, cells A (${plan.cells.a.dx},${plan.cells.a.dz}) B (${plan.cells.b.dx},${plan.cells.b.dz}), ` +
        `sweeps ${plan.sweeps.map((s) => `${s.sweep_deg.toFixed(0)}deg@${s.omega_deg_per_sec}`).join(" / ")}`,
    );

    let scene = null;
    let attempts = 0;
    const hops = [];
    while (scene === null && attempts < SCENE_MAX_ATTEMPTS) {
      if (attempts > 0) {
        hops.push(await hop(bot, rcon, botPosition, attempts, rng));
        await sleep(SCENE_HOP_SETTLE_MS);
      }
      attempts++;
      const limits = attempts >= RELAXED_FROM_ATTEMPT ? TERRAIN_RELAXED : TERRAIN_STRICT;
      scene = await pickArena(bot, limits);
      if (scene) scene.terrain.limits = limits;
    }
    this._episodeMetadata.scene_search_attempts = attempts;
    this._episodeMetadata.scene_hops = hops;
    if (scene === null) return null;
    const { centre, arena } = scene;
    console.log(
      `[${bot.username}] arena: centre (${centre.x}, ${centre.y}, ${centre.z}) terrain ${JSON.stringify(scene.terrain)} after ${attempts} attempt(s)`,
    );

    for (const cmd of ["gamerule doTileDrops false", "gamerule doMobLoot false", "gamerule doMobSpawning false"]) {
      try {
        await rcon.send(cmd);
      } catch (err) {
        console.log(`[${bot.username}] ${cmd} failed: ${err.message}`);
      }
    }
    const clearHeight = plan.shape.height + 1;
    const terra = await terraform(bot, rcon, arena, clearHeight);
    const terraformInfo = { ...terra, terrain_before: scene.terrain };
    terraformInfo.repair_rounds = await repairArena(bot, rcon, arena, terra.top_block, clearHeight);
    await sleep(500);

    // Bot positions and looks.
    const posOf = (c) => [centre.x + c.dx, centre.y, centre.z + c.dz];
    const posA = posOf(plan.cells.a);
    const posB = posOf(plan.cells.b);
    const eyeA = eyeOf({ x: posA[0], y: posA[1], z: posA[2] });
    const eyeB = eyeOf({ x: posB[0], y: posB[1], z: posB[2] });
    const hitA = wallHit(centre, eyeA, plan.yaw0A);
    // B looks at A's wall point shifted along that wall, with a yaw jitter.
    const alongX = hitA.wall === "north" || hitA.wall === "south";
    const target = {
      x: hitA.x + (alongX ? plan.overlap_shift : 0),
      z: hitA.z + (alongX ? 0 : plan.overlap_shift),
    };
    const yaw0B = wrapRad(yawTo(eyeB, target) + plan.overlap_jitter);
    const hitB = wallHit(centre, eyeB, yaw0B);
    const names = [bot.username, args.other_bot_name];
    const bots = {};
    [[names[0], posA, plan.yaw0A, plan.pitch0[0], plan.sweeps[0], hitA],
      [names[1], posB, yaw0B, plan.pitch0[1], plan.sweeps[1], hitB]].forEach(([name, pos, yaw0, pitch0, sw, hit]) => {
      bots[name] = {
        name,
        pos,
        yaw0,
        pitch0,
        sweep_deg: sw.sweep_deg,
        omega_deg_per_sec: sw.omega_deg_per_sec,
        direction: sw.direction,
        pitch_amp_deg: sw.pitch_amp_deg,
        pitch_period_s: sw.pitch_period_s,
        pitch_phase: sw.pitch_phase,
        wall_hit: hit ? { x: hit.x, z: hit.z, wall: hit.wall } : null,
      };
    });
    const dirA = viewDir(plan.yaw0A);
    const dirB = viewDir(yaw0B);
    const startOverlap = {
      view_angle_deg: Math.acos(Math.max(-1, Math.min(1, dirA.x * dirB.x + dirA.z * dirB.z))) / DEG,
      wall_a: hitA ? hitA.wall : null,
      wall_b: hitB ? hitB.wall : null,
      same_or_adjacent_wall: hitA && hitB ? wallsAdjacent(hitA.wall, hitB.wall) : false,
      bot_distance: Math.hypot(posA[0] - posB[0], posA[2] - posB[2]),
    };

    for (const b of Object.values(bots)) {
      const res = await rcon.send(`tp ${b.name} ${b.pos[0]} ${b.pos[1]} ${b.pos[2]}`);
      console.log(`[${bot.username}] tp ${b.name} -> ${String(res).trim()}`);
    }
    await sleep(1500);
    const mine = bots[bot.username].pos;
    if (bot.entity.position.distanceTo(new Vec3(mine[0], mine[1], mine[2])) > 0.3) {
      console.log(`[${bot.username}] primary not at start (${bot.entity.position}), giving up`);
      return null;
    }
    for (const b of Object.values(bots)) {
      if (!isStandable(bot, b.pos[0], b.pos[1], b.pos[2])) {
        console.log(`[${bot.username}] start of ${b.name} not standable after terraform, giving up`);
        return null;
      }
    }

    const cx = Math.floor(centre.x);
    const cz = Math.floor(centre.z);
    const killCmd = `execute positioned ${cx} ${centre.y} ${cz} run kill @e[type=!minecraft:player,distance=..${MOB_CLEAR_RADIUS}]`;
    try {
      await rcon.send(killCmd);
    } catch (err) {
      console.log(`[${bot.username}] ${killCmd} failed: ${err.message}`);
    }
    const blocks = plan.shape.blocks.map((b) => ({ x: cx + b.dx, y: centre.y + b.dy, z: cz + b.dz, block: b.block, part: b.part }));
    const build = await buildEnclosure(bot, rcon, blocks, rng);
    this._episodeMetadata.enclosure_build = build;
    if (!build.ok) return null;
    try {
      await rcon.send(killCmd);
    } catch (err) {
      console.log(`[${bot.username}] second ${killCmd} failed: ${err.message}`);
    }
    await sleep(500);

    return {
      centre,
      arena,
      terraform: terraformInfo,
      enclosure: {
        kind: "square_enclosure_random_concrete",
        inner_size: INNER_SIZE,
        wall_height: plan.shape.height,
        wall_offset: WALL_OFFSET,
        box_min: [cx - WALL_OFFSET, centre.y - 1, cz - WALL_OFFSET],
        box_max: [cx + WALL_OFFSET, centre.y + plan.shape.height - 1, cz + WALL_OFFSET],
        num_blocks: blocks.length,
        blocks,
      },
      bots,
      start_overlap: startOverlap,
      attempts,
      hops,
    };
  }

  async entryPoint(bot, rcon, sharedBotRng, coordinator, iterationID, episodeNum, args) {
    coordinator.onceEvent(
      "sharedArenaPhase",
      episodeNum,
      getOnSharedArenaPhaseFn(bot, rcon, sharedBotRng, coordinator, episodeNum, this, args),
    );
    coordinator.sendToOtherBot("sharedArenaPhase", bot.entity.position.clone(), episodeNum, "entryPoint end");
  }

  async tearDownEpisode(bot, rcon, sharedBotRng, coordinator, episodeNum, args) {
    if (!this._isPrimary) return;
    for (const cmd of ["gamerule doTileDrops true", "gamerule doMobLoot true", "gamerule doMobSpawning true"]) {
      try {
        await rcon.send(cmd);
      } catch (err) {
        console.log(`[${bot.username}] teardown ${cmd} failed: ${err.message}`);
      }
    }
    if (this._layout) {
      const e = this._layout.enclosure;
      const top = (this._layout.terraform && this._layout.terraform.top_block) || "grass_block";
      try {
        await rcon.send(`fill ${e.box_min[0]} ${e.box_min[1] + 1} ${e.box_min[2]} ${e.box_max[0]} ${e.box_max[1]} ${e.box_max[2]} minecraft:air`);
        await rcon.send(`fill ${e.box_min[0]} ${e.box_min[1]} ${e.box_min[2]} ${e.box_max[0]} ${e.box_min[1]} ${e.box_max[2]} minecraft:${top}`);
      } catch (err) {
        console.log(`[${bot.username}] enclosure cleanup failed: ${err.message}`);
      }
    }
  }
}

module.exports = { SharedArenaEvalEpisode };

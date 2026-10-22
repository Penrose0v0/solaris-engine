const seedrandom = require("seedrandom");
const { Vec3 } = require("vec3");

const { stopAll } = require("../../primitives/movement");
const { decidePrimaryBot } = require("../../utils/coordination");
const { sleep } = require("../../utils/helpers");
const { BaseEpisode } = require("../base-episode");
const { sceneUtils } = require("./traj-segments-eval-episode");

const { isPassable, isStandable, isFlatGround, blockAtXYZ, EYE_HEIGHT } = sceneUtils;

// Fixed base seed; the per-episode seed is derived from (base seed, instance
// id, episode number) so the plan is reproducible.
const CO_OBSERVE_BASE_SEED = 20261020;

// Colour-contrasting concrete for the landmark columns.
const BLOCK_TYPES = [
  "red_concrete",
  "blue_concrete",
  "yellow_concrete",
  "white_concrete",
  "black_concrete",
  "orange_concrete",
  "lime_concrete",
  "magenta_concrete",
  "cyan_concrete",
  "purple_concrete",
];

// Landmark: an asymmetric structure drawn from the episode seed. Columns on
// a (2*LANDMARK_HALF+1)^2 footprint with random heights 1..LANDMARK_MAX_HEIGHT
// (at least three different heights, one column at full height), plus
// LANDMARK_PROTRUSIONS blocks sticking out sideways above ground level, at
// most LANDMARK_MAX_BLOCKS blocks in all (the 3x3x4 box holds 36) so most of
// the background stays visible. Every block gets its own random colour
// (never the colour of the block below it).
const LANDMARK_HALF = 1;
const LANDMARK_MAX_HEIGHT = 4;
const LANDMARK_MIN_DISTINCT_HEIGHTS = 3;
const LANDMARK_PROTRUSIONS = [1, 2];
const LANDMARK_MAX_BLOCKS = 28;
// Height above the patch level both bots aim at (lower-middle of the
// landmark, so the ground around it stays in the lower half of the frame).
const AIM_HEIGHT = 1.0;

// Motion (physics ticks; recording ~20 fps so 1 tick ~ 1 frame). Both bots
// stand still for LEAD_TICKS, circle-strafe for MOTION_TICKS (the eval
// window) and stand still again for TAIL_TICKS.
const LEAD_TICKS = 10;
const MOTION_TICKS = 160;
// (The tail is long enough for ~10 fully static frames after the bots have
// coasted to a stop: the FOV check in postprocess uses them.)
const TAIL_TICKS = 25;

// Geometry. Radii are horizontal distances to the patch centre. Both bots
// orbit the same centre; in "same" mode their radii differ by at most
// SAME_MAX_DR, in "opposite" mode by OPPOSITE_DR (so they pass each other
// without touching).
const RADIUS_RANGE = [6, 12];
const RADIUS_STEP = 0.5;
const SAME_MAX_DR = 2.0;
const OPPOSITE_DR = [1.5, 3.0];
// Initial angle between the two lines of sight to the centre, and whether
// the opposite direction mode is used (episodes alternate same/opposite
// when it is, else all are same-direction). prod001 was recorded with
// [30, 90] and both modes; the eval side could score almost only
// same-direction episodes that started at 30-50 deg, hence prod002's
// [25, 50] / same only.
const VIEW_ANGLE_RANGE_DEG = [25, 50];
const OPPOSITE_MODE_ENABLED = false;
// Radius controller: strafing alone lets the radius creep outwards, so
// forward/back is pressed when the predicted radius leaves the dead band.
const RADIUS_DEADBAND = 0.1;
const RADIUS_PREDICT_TICKS = 4;
// Arena: the square of half-size ceil(max radius + RING_HALF_WIDTH +
// DISK_MARGIN) around the centre is levelled through rcon before recording
// (nothing above the ground level, ground filled where open). Candidate
// centres are rated on the natural terrain of that square first, so the
// levelling only ever trims a few blocks: every column's surface within
// MAX_SURFACE_OFFSET of the ground level, most columns (TERRAIN_STRICT /
// TERRAIN_RELAXED .level) within one block of it, few raised columns
// (trees, slopes) and little water.
const RING_HALF_WIDTH = 0.75;
const DISK_MARGIN = 1.0;
const TERRAFORM_CLEAR_HEIGHT = 5;
const SURFACE_SCAN = 16;
const MAX_SURFACE_OFFSET = 3;
// `odd`: columns whose surface is not plain full-cube ground (tree canopy,
// ice, snow layers, slabs, paths); a frozen lake or a forest roof can look
// perfectly level otherwise.
const TERRAIN_STRICT = { level: 0.8, raised: 0.08, liquid: 0.1, odd: 0.1 };
const TERRAIN_RELAXED = { level: 0.65, raised: 0.15, liquid: 0.1, odd: 0.15 };
const ARENA_REPAIR_ROUNDS = 2;
const ARENA_REPAIR_SETTLE_MS = 1200;
const RELAXED_FROM_ATTEMPT = 9;
const MAX_UNLOADED_FRACTION = 0.3;
const HEIGHT_MAP_RETRIES = 6;
const HEIGHT_MAP_RETRY_MS = 2000;
const MAX_VIEW_CANDIDATES = 10;
// Scene hops: odd attempts jump to the nearest flat biome of a random point
// within the hop range (`locate biome`), even attempts to a random spot.
const FLAT_BIOMES = ["plains", "savanna", "desert", "sunflower_plains", "meadow"];
const HOP_SPREAD_RADIUS = 24;
// View check: nothing close in front of the camera (same value as
// sceneUtils.isViewGood); the horizon may be empty, the concrete landmark
// is the scene anchor.
const VIEW_MIN_CLEAR_BLOCKS = 6;
const VIEW_LANDMARK_RANGE_BLOCKS = 40;
const DIRT_TOPS = ["grass_block", "dirt", "coarse_dirt", "podzol", "mycelium", "rooted_dirt", "mud"];
// Mobs are removed this far around the centre before recording.
const MOB_CLEAR_RADIUS = 48;

// Scene search (see pickScene).
const SCENE_SEARCH_RADIUS = 8;
const SCENE_SEARCH_BUDGET_MS = 6000;
const SCENE_MAX_ATTEMPTS = 16;
const SCENE_HOP_RANGE = 32;
const SCENE_HOP_MAX_RANGE = 400;
const SCENE_HOP_SETTLE_MS = 4000;
const BEARING_CANDIDATES = 12;
const VIEW_CHECK_BEARINGS = 4;
const SECONDARY_CHUNK_SETTLE_MS = 2500;

const DEG = Math.PI / 180;

// ---------------------------------------------------------------------------
// Geometry helpers (mineflayer convention: yaw 0 = north = -z, yaw grows
// counter-clockwise, pitch positive = up). Bearings are measured in the x-z
// plane: bearing = atan2(z - cz, x - cx); strafing LEFT while facing the
// centre increases the bearing, RIGHT decreases it.
// ---------------------------------------------------------------------------

function eyeOf(pos) {
  return { x: pos.x, y: pos.y + EYE_HEIGHT, z: pos.z };
}

function anglesTo(eye, point) {
  const dx = point.x - eye.x;
  const dy = point.y - eye.y;
  const dz = point.z - eye.z;
  return {
    yaw: Math.atan2(-dx, -dz),
    pitch: Math.atan2(dy, Math.hypot(dx, dz)),
  };
}

function wrapRad(a) {
  return ((((a + Math.PI) % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI)) - Math.PI;
}

function onCircle(centre, r, bearing) {
  return { x: centre.x + r * Math.cos(bearing), y: centre.y, z: centre.z + r * Math.sin(bearing) };
}

function bearingOf(centre, pos) {
  return Math.atan2(pos.z - centre.z, pos.x - centre.x);
}

function shuffled(arr, rng) {
  const a = arr.slice();
  for (let k = a.length - 1; k > 0; k--) {
    const j = Math.floor(rng() * (k + 1));
    [a[k], a[j]] = [a[j], a[k]];
  }
  return a;
}

function roundTo(v, step) {
  return Math.round(v / step) * step;
}

function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

// ---------------------------------------------------------------------------
// Plan
// ---------------------------------------------------------------------------

/**
 * Per-episode plan: direction mode (alternating by episode number), strafe
 * keys, initial view angle and the wanted radii. In opposite mode the bots
 * start moving apart, so the first part of the window keeps the angle near
 * its initial value and the pass-by comes late.
 */
function buildPlan(rng, episodeNum) {
  const mode = OPPOSITE_MODE_ENABLED && episodeNum % 2 === 1 ? "opposite" : "same";
  const keyA = rng() < 0.5 ? "left" : "right";
  const keyB = mode === "same" ? keyA : keyA === "left" ? "right" : "left";
  const theta0 =
    VIEW_ANGLE_RANGE_DEG[0] + rng() * (VIEW_ANGLE_RANGE_DEG[1] - VIEW_ANGLE_RANGE_DEG[0]);
  let side;
  if (mode === "same") {
    side = rng() < 0.5 ? 1 : -1;
  } else {
    // A left (bearing increasing) -> B below A moving right: they separate.
    side = keyA === "left" ? -1 : 1;
  }
  const rA = roundTo(RADIUS_RANGE[0] + rng() * (RADIUS_RANGE[1] - RADIUS_RANGE[0]), RADIUS_STEP);
  let rB;
  if (mode === "same") {
    rB = rA + (rng() * 2 - 1) * SAME_MAX_DR;
  } else {
    const dr = OPPOSITE_DR[0] + rng() * (OPPOSITE_DR[1] - OPPOSITE_DR[0]);
    const up = rA + dr <= RADIUS_RANGE[1];
    const down = rA - dr >= RADIUS_RANGE[0];
    rB = up && down ? (rng() < 0.5 ? rA + dr : rA - dr) : up ? rA + dr : rA - dr;
  }
  rB = roundTo(clamp(rB, RADIUS_RANGE[0], RADIUS_RANGE[1]), RADIUS_STEP);
  const shape = buildLandmarkShape(rng);
  const bearingOffset = rng() * 2 * Math.PI;
  return { mode, keyA, keyB, theta0, side, rA, rB, shape, bearingOffset };
}

/**
 * Landmark shape relative to the centre column (dx, dy, dz; dy 0 = ground
 * level): column heights, protruding blocks and a colour per block. See the
 * LANDMARK_* constants.
 */
function buildLandmarkShape(rng) {
  const side = 2 * LANDMARK_HALF + 1;
  for (;;) {
    const heights = [];
    for (let i = 0; i < side * side; i++) heights.push(1 + Math.floor(rng() * LANDMARK_MAX_HEIGHT));
    if (new Set(heights).size < LANDMARK_MIN_DISTINCT_HEIGHTS) continue;
    if (!heights.includes(LANDMARK_MAX_HEIGHT)) continue;
    const cells = new Set();
    const blocks = [];
    const columns = [];
    heights.forEach((h, i) => {
      const dx = Math.floor(i / side) - LANDMARK_HALF;
      const dz = (i % side) - LANDMARK_HALF;
      columns.push({ dx, dz, height: h });
      for (let dy = 0; dy < h; dy++) {
        blocks.push({ dx, dy, dz });
        cells.add(`${dx},${dy},${dz}`);
      }
    });
    const nProt =
      LANDMARK_PROTRUSIONS[0] +
      Math.floor(rng() * (LANDMARK_PROTRUSIONS[1] - LANDMARK_PROTRUSIONS[0] + 1));
    const protrusions = [];
    for (let tries = 0; protrusions.length < nProt && tries < 40; tries++) {
      const col = columns[Math.floor(rng() * columns.length)];
      if (col.height < 2) continue;
      const dirs = [];
      if (col.dx === -LANDMARK_HALF) dirs.push([-1, 0]);
      if (col.dx === LANDMARK_HALF) dirs.push([1, 0]);
      if (col.dz === -LANDMARK_HALF) dirs.push([0, -1]);
      if (col.dz === LANDMARK_HALF) dirs.push([0, 1]);
      if (dirs.length === 0) continue;
      const [ex, ez] = dirs[Math.floor(rng() * dirs.length)];
      const dy = 1 + Math.floor(rng() * (col.height - 1)); // above ground, beside the column
      const b = { dx: col.dx + ex, dy, dz: col.dz + ez };
      const key = `${b.dx},${b.dy},${b.dz}`;
      if (cells.has(key)) continue;
      cells.add(key);
      blocks.push(b);
      protrusions.push(b);
    }
    if (blocks.length > LANDMARK_MAX_BLOCKS) continue;
    const colourAt = new Map();
    for (const b of blocks) {
      const below = colourAt.get(`${b.dx},${b.dy - 1},${b.dz}`);
      let c;
      do {
        c = BLOCK_TYPES[Math.floor(rng() * BLOCK_TYPES.length)];
      } while (c === below);
      b.block = c;
      colourAt.set(`${b.dx},${b.dy},${b.dz}`, c);
    }
    return { columns, protrusions, blocks };
  }
}

// ---------------------------------------------------------------------------
// Scene selection
// ---------------------------------------------------------------------------

function isLiquidBlock(block) {
  return block !== null && ["water", "lava", "bubble_column"].includes(block.name);
}

/**
 * Surface map of the square base +- half: per column the y of the topmost
 * non-passable block within base.y +- SURFACE_SCAN (liquids count as
 * surface). An unloaded column maps to null, a column whose surface is
 * outside the scan range to {h: null}.
 */
function surfaceMap(bot, base, half) {
  const map = new Map();
  let unloaded = 0;
  for (let dx = -half; dx <= half; dx++) {
    for (let dz = -half; dz <= half; dz++) {
      const x = base.x + dx;
      const z = base.z + dz;
      let entry = null;
      if (blockAtXYZ(bot, x, base.y, z) !== null) {
        entry = { h: null, liquid: false, odd: false };
        for (let y = base.y + SURFACE_SCAN; y >= base.y - SURFACE_SCAN; y--) {
          const b = blockAtXYZ(bot, x, y, z);
          if (b === null) break;
          if (!isPassable(b)) {
            const liquid = isLiquidBlock(b);
            entry = { h: y, liquid, odd: !liquid && !isFlatGround(b) };
            break;
          }
        }
      }
      if (entry === null) unloaded++;
      map.set(`${dx},${dz}`, entry);
    }
  }
  return { map, unloaded, total: (2 * half + 1) ** 2 };
}

/**
 * Rate the square of half-size `half` around column (dx,dz) with ground
 * level g (surface y): returns {stats} or {reason} when the square cannot
 * be levelled gently under `limits`.
 */
function rateArena(surface, dx0, dz0, g, half, limits) {
  let n = 0;
  let level = 0;
  let raised = 0;
  let liquid = 0;
  let odd = 0;
  let maxOff = 0;
  let minOff = 0;
  for (let dx = -half; dx <= half; dx++) {
    for (let dz = -half; dz <= half; dz++) {
      const e = surface.map.get(`${dx0 + dx},${dz0 + dz}`);
      if (e === undefined || e === null) return { reason: "unloaded" };
      if (e.h === null) return { reason: "cliff" };
      const o = e.h - g;
      if (Math.abs(o) > MAX_SURFACE_OFFSET) return { reason: "cliff" };
      n++;
      if (e.liquid) liquid++;
      if (e.odd) odd++;
      if (Math.abs(o) <= 1) level++;
      if (o >= 2) raised++;
      maxOff = Math.max(maxOff, o);
      minOff = Math.min(minOff, o);
    }
  }
  const stats = {
    columns: n,
    level_fraction: level / n,
    raised_fraction: raised / n,
    liquid_fraction: liquid / n,
    odd_fraction: odd / n,
    max_offset: maxOff,
    min_offset: minOff,
  };
  if (stats.liquid_fraction > limits.liquid) return { reason: "water" };
  if (stats.odd_fraction > limits.odd) return { reason: "odd" };
  if (stats.level_fraction < limits.level) return { reason: "rough" };
  if (stats.raised_fraction > limits.raised) return { reason: "raised" };
  return { stats };
}

// Distance to the first solid block along a ray, treating the arena volume
// (which will be cleared) as air. Infinity when nothing is hit (also when
// the ray leaves the loaded chunks).
function rayHitDistanceArena(bot, eye, yaw, pitch, maxDist, arena) {
  const ch = Math.cos(pitch);
  const dx = -Math.sin(yaw) * ch;
  const dy = Math.sin(pitch);
  const dz = -Math.cos(yaw) * ch;
  for (let t = 0.5; t <= maxDist; t += 0.5) {
    const px = eye.x + dx * t;
    const py = eye.y + dy * t;
    const pz = eye.z + dz * t;
    const inArena =
      Math.abs(Math.floor(px) - arena.cx) <= arena.half &&
      Math.abs(Math.floor(pz) - arena.cz) <= arena.half &&
      py >= arena.y &&
      py < arena.y + TERRAFORM_CLEAR_HEIGHT;
    if (inArena) continue;
    const block = blockAtXYZ(bot, px, py, pz);
    if (block === null) return Infinity;
    if (block.boundingBox === "block") return t;
  }
  return Infinity;
}

// View from `pos` towards the centre on the levelled arena: nothing close
// around the view centre. Also counts the rays (of 21 at pitch 0..12 deg)
// that end on terrain within VIEW_LANDMARK_RANGE_BLOCKS, for the record.
function viewFromArena(bot, pos, yaw, arena) {
  const eye = eyeOf(pos);
  for (const dyaw of [-20, 0, 20]) {
    const d = rayHitDistanceArena(bot, eye, yaw + dyaw * DEG, 0, VIEW_MIN_CLEAR_BLOCKS, arena);
    if (d < VIEW_MIN_CLEAR_BLOCKS) return { ok: false, horizonRays: 0 };
  }
  let horizonRays = 0;
  for (const dpitch of [0, 6, 12]) {
    for (let dyaw = -30; dyaw <= 30; dyaw += 10) {
      const d = rayHitDistanceArena(
        bot,
        eye,
        yaw + dyaw * DEG,
        dpitch * DEG,
        VIEW_LANDMARK_RANGE_BLOCKS,
        arena,
      );
      if (d !== Infinity) horizonRays++;
    }
  }
  return { ok: true, horizonRays };
}

// Views from VIEW_CHECK_BEARINGS points of the circle towards the centre;
// returns the summed horizon-ray count or -1 when a view is blocked.
function circleViews(bot, centre, r, bearing0, arena) {
  let rays = 0;
  for (let i = 0; i < VIEW_CHECK_BEARINGS; i++) {
    const b = bearing0 + (2 * Math.PI * i) / VIEW_CHECK_BEARINGS;
    const p = onCircle(centre, r, b);
    const yaw = anglesTo(eyeOf(p), centre).yaw;
    const v = viewFromArena(bot, p, yaw, arena);
    if (!v.ok) return -1;
    rays += v.horizonRays;
  }
  return rays;
}

/**
 * Search the primary bot's surroundings for an arena centre: a square that
 * is nearly level already (see rateArena), rated best-first, with both start
 * points `theta0` apart on their circles and a landmark-rich view. The
 * arena is levelled afterwards by terraform().
 * @returns {{centre, arena, terrain, bearingA, bearingB, startA, startB, candidates, rejected}|null}
 */
async function pickScene(bot, plan, rng, limits) {
  const base = bot.entity.position.floored();
  const half = Math.ceil(Math.max(plan.rA, plan.rB) + RING_HALF_WIDTH + DISK_MARGIN);
  let surface = null;
  for (let i = 0; i <= HEIGHT_MAP_RETRIES; i++) {
    surface = surfaceMap(bot, base, SCENE_SEARCH_RADIUS + half);
    if (surface.unloaded / surface.total <= MAX_UNLOADED_FRACTION) break;
    console.log(
      `[${bot.username}] surface map: ${surface.unloaded}/${surface.total} columns unloaded, waiting for chunks`,
    );
    await sleep(HEIGHT_MAP_RETRY_MS);
  }

  const rejected = { centre: 0, unloaded: 0, cliff: 0, water: 0, odd: 0, rough: 0, raised: 0, view: 0 };
  const rated = [];
  for (let dx = -SCENE_SEARCH_RADIUS; dx <= SCENE_SEARCH_RADIUS; dx++) {
    for (let dz = -SCENE_SEARCH_RADIUS; dz <= SCENE_SEARCH_RADIUS; dz++) {
      const e = surface.map.get(`${dx},${dz}`);
      if (!e || e.h === null || e.liquid || e.odd) {
        rejected.centre++;
        continue;
      }
      const r = rateArena(surface, dx, dz, e.h, half, limits);
      if (r.reason) {
        rejected[r.reason]++;
        continue;
      }
      rated.push({ dx, dz, g: e.h, stats: r.stats, d2: dx * dx + dz * dz });
    }
  }
  rated.sort(
    (a, b) =>
      b.stats.level_fraction - a.stats.level_fraction ||
      a.stats.raised_fraction - b.stats.raised_fraction ||
      a.d2 - b.d2,
  );

  const deadline = Date.now() + SCENE_SEARCH_BUDGET_MS;
  let candidates = 0;
  for (const cand of rated.slice(0, MAX_VIEW_CANDIDATES)) {
    if (Date.now() > deadline) break;
    candidates++;
    const cx = base.x + cand.dx;
    const cz = base.z + cand.dz;
    const y = cand.g + 1;
    const centre = { x: cx + 0.5, y, z: cz + 0.5 };
    const arena = { cx, cz, y, half };
    const offsets = shuffled(
      Array.from({ length: BEARING_CANDIDATES }, (_, k) => (2 * Math.PI * k) / BEARING_CANDIDATES),
      rng,
    );
    for (const off of offsets) {
      const bearingA = wrapRad(plan.bearingOffset + off);
      const bearingB = wrapRad(bearingA + plan.side * plan.theta0 * DEG);
      const raysA = circleViews(bot, centre, plan.rA, bearingA, arena);
      if (raysA < 0) continue;
      const raysB = circleViews(bot, centre, plan.rB, bearingB, arena);
      if (raysB < 0) continue;
      return {
        centre,
        arena,
        terrain: { ...cand.stats, horizon_rays: raysA + raysB },
        bearingA,
        bearingB,
        startA: onCircle(centre, plan.rA, bearingA),
        startB: onCircle(centre, plan.rB, bearingB),
        candidates,
        rejected,
      };
    }
    rejected.view++;
    await sleep(0);
  }
  console.log(
    `[${bot.username}] scene search: no suitable arena among ${rated.length} rated centres ` +
      `(${candidates} view-checked), rejected by ${JSON.stringify(rejected)}, unloaded columns ${surface.unloaded}/${surface.total}`,
  );
  return null;
}

/**
 * Move the primary bot to a new search spot: on odd attempts the nearest
 * flat biome (`locate biome`) of a random point within `range` of the
 * original position, on even attempts a random spot within `range`; the
 * bot lands on the surface within HOP_SPREAD_RADIUS of that point.
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
    // No surface to land on (ocean): pick another random point.
    const a = rng() * 2 * Math.PI;
    x = Math.floor(origin.x + range * Math.cos(a));
    z = Math.floor(origin.z + range * Math.sin(a));
    how += `; retry at ${x},${z}`;
  }
  console.log(`[${bot.username}] scene hop ${attempt}: ${how} -> ${res}`);
  return how;
}

/**
 * Walk the levelled square through this client's block data and fix every
 * column the fills could not express: ground that is not a plain full cube
 * (a flower or sapling sitting in a hole, ice, a slab, a path block) gets
 * the top block, anything left above the ground level is cleared. Returns
 * the number of columns touched per round.
 */
async function repairArena(bot, rcon, arena, top) {
  const { cx, cz, y, half } = arena;
  const rounds = [];
  for (let round = 0; round < ARENA_REPAIR_ROUNDS; round++) {
    await sleep(ARENA_REPAIR_SETTLE_MS);
    let ground = 0;
    let above = 0;
    let unloaded = 0;
    for (let x = cx - half; x <= cx + half; x++) {
      for (let z = cz - half; z <= cz + half; z++) {
        const g = blockAtXYZ(bot, x, y - 1, z);
        if (g === null) {
          unloaded++;
          continue;
        }
        if (!isFlatGround(g)) {
          await rcon.send(`setblock ${x} ${y - 1} ${z} minecraft:${top}`);
          ground++;
        }
        for (let dy = 0; dy < TERRAFORM_CLEAR_HEIGHT; dy++) {
          const b = blockAtXYZ(bot, x, y + dy, z);
          if (b !== null && b.name !== "air") {
            await rcon.send(`fill ${x} ${y} ${z} ${x} ${y + TERRAFORM_CLEAR_HEIGHT - 1} ${z} minecraft:air`);
            above++;
            break;
          }
        }
      }
    }
    rounds.push({ ground, above, unloaded });
    console.log(`[${bot.username}] arena repair round ${round}: ${JSON.stringify(rounds[round])}`);
    if (ground + above === 0) break;
  }
  return rounds;
}

// Make a start spot standable when the levelling left an odd block there
// (non-cube ground such as a path block or slab, or a straggler above).
async function repairSpot(bot, rcon, pos, arena, top) {
  const x = Math.floor(pos.x);
  const z = Math.floor(pos.z);
  const y = arena.y;
  for (const cmd of [
    `setblock ${x} ${y - 1} ${z} minecraft:${top}`,
    `fill ${x} ${y} ${z} ${x} ${y + 2} ${z} minecraft:air`,
  ]) {
    const res = await rcon.send(cmd);
    console.log(`[${bot.username}] repair ${cmd} -> ${String(res).trim()}`);
  }
}

/**
 * Level the arena through rcon: clear everything from the ground level up,
 * then fill the ground layer and the one below where they are open (holes,
 * water, plants, leaves). Returns the commands' results.
 */
async function terraform(bot, rcon, arena) {
  const { cx, cz, y, half } = arena;
  const x0 = cx - half;
  const x1 = cx + half;
  const z0 = cz - half;
  const z1 = cz + half;
  const topBlockAt = blockAtXYZ(bot, cx, y - 1, cz);
  const top = topBlockAt && isFlatGround(topBlockAt) ? topBlockAt.name : "grass_block";
  const filler = DIRT_TOPS.includes(top) ? "dirt" : "stone";
  const cmds = [
    `fill ${x0} ${y} ${z0} ${x1} ${y + TERRAFORM_CLEAR_HEIGHT - 1} ${z1} minecraft:air`,
    `fill ${x0} ${y - 2} ${z0} ${x1} ${y - 2} ${z1} minecraft:${filler} replace #minecraft:replaceable`,
    `fill ${x0} ${y - 2} ${z0} ${x1} ${y - 2} ${z1} minecraft:${filler} replace #minecraft:leaves`,
    `fill ${x0} ${y - 1} ${z0} ${x1} ${y - 1} ${z1} minecraft:${top} replace #minecraft:replaceable`,
    `fill ${x0} ${y - 1} ${z0} ${x1} ${y - 1} ${z1} minecraft:${top} replace #minecraft:leaves`,
  ];
  // Columns that were higher than the ground level expose the layer under
  // their old surface (dirt/stone); cover it with the top block.
  for (const under of ["dirt", "coarse_dirt", "stone"]) {
    if (under !== top) {
      cmds.push(`fill ${x0} ${y - 1} ${z0} ${x1} ${y - 1} ${z1} minecraft:${top} replace minecraft:${under}`);
    }
  }
  const results = [];
  for (const cmd of cmds) {
    try {
      const res = String(await rcon.send(cmd)).trim();
      console.log(`[${bot.username}] ${cmd} -> ${res}`);
      results.push(res);
    } catch (err) {
      console.log(`[${bot.username}] ${cmd} failed: ${err.message}`);
      results.push(`error: ${err.message}`);
    }
  }
  return { top_block: top, filler_block: filler, box: [x0, y - 2, z0, x1, y + TERRAFORM_CLEAR_HEIGHT - 1, z1], results };
}

// ---------------------------------------------------------------------------
// RCON helpers
// ---------------------------------------------------------------------------

async function rconBlockIs(rcon, pos, blockName) {
  const res = await rcon.send(
    `execute if block ${pos[0]} ${pos[1]} ${pos[2]} minecraft:${blockName}`,
  );
  return /passed/i.test(String(res));
}

// Recipes, chat and advancement notifications: unlock every recipe for the
// bot and its camera so no "new recipe" toast can pop, silence advancement
// chat and command feedback. Once per bot process.
async function quietNotificationsOnce(bot, rcon) {
  if (bot._coObserveQuietDone) return;
  bot._coObserveQuietDone = true;
  for (const cmd of [
    `recipe give ${bot.username} *`,
    `recipe give Camera${bot.username} *`,
    "gamerule announceAdvancements false",
    "gamerule sendCommandFeedback false",
    "gamerule logAdminCommands false",
  ]) {
    try {
      const res = await rcon.send(cmd);
      console.log(`[${bot.username}] ${cmd} -> ${String(res).trim()}`);
    } catch (err) {
      console.log(`[${bot.username}] ${cmd} failed: ${err.message}`);
    }
  }
}

// Landmark blocks in world coordinates (ground level = centre.y).
function landmarkBlocks(centre, shape) {
  const cx = Math.floor(centre.x);
  const cz = Math.floor(centre.z);
  return shape.blocks.map((b) => ({ x: cx + b.dx, y: centre.y + b.dy, z: cz + b.dz, block: b.block }));
}

function landmarkBounds(blocks) {
  const xs = blocks.map((b) => b.x);
  const ys = blocks.map((b) => b.y);
  const zs = blocks.map((b) => b.z);
  return {
    min: [Math.min(...xs), Math.min(...ys), Math.min(...zs)],
    max: [Math.max(...xs), Math.max(...ys), Math.max(...zs)],
  };
}

// Build the landmark through rcon and wait until the server and this client
// both see every block. Returns true when all of them are confirmed.
async function buildLandmark(bot, rcon, blocks) {
  for (const b of blocks) {
    const res = await rcon.send(`setblock ${b.x} ${b.y} ${b.z} minecraft:${b.block}`);
    if (!/Changed/i.test(String(res))) {
      console.log(`[${bot.username}] landmark setblock ${b.x},${b.y},${b.z} ${b.block} -> ${String(res).trim()}`);
    }
  }
  let serverOk = true;
  for (const b of blocks) {
    if (!(await rconBlockIs(rcon, [b.x, b.y, b.z], b.block))) serverOk = false;
  }
  const deadline = Date.now() + 4000;
  let clientOk = false;
  while (!clientOk && Date.now() < deadline) {
    clientOk = blocks.every((b) => {
      const c = bot.blockAt(new Vec3(b.x, b.y, b.z));
      return c && c.name === b.block;
    });
    if (!clientOk) await sleep(100);
  }
  console.log(`[${bot.username}] landmark ${blocks.length} blocks server_ok=${serverOk} client_ok=${clientOk}`);
  return serverOk && clientOk;
}

// ---------------------------------------------------------------------------
// Coordination helper
// ---------------------------------------------------------------------------

/**
 * Rendezvous on `name`: register, send, and re-send once from the handler so
 * the slower bot's listener is always triggered (double-send pattern).
 */
function rendezvous(coordinator, name, episodeNum) {
  return new Promise((resolve) => {
    coordinator.onceEvent(name, episodeNum, () => {
      coordinator.sendToOtherBot(name, {}, episodeNum, `${name} handler`);
      resolve();
    });
    coordinator.sendToOtherBot(name, {}, episodeNum, `${name} start`);
  });
}

// ---------------------------------------------------------------------------
// Motion
// ---------------------------------------------------------------------------

/**
 * Circle-strafe for MOTION_TICKS: every tick the bot looks at the aim point
 * and holds `key` (left/right); forward/back is pressed while the radius,
 * predicted RADIUS_PREDICT_TICKS ahead from the current velocity, sits
 * outside the dead band around `r0`. Returns the per-tick trace.
 */
async function circleStrafe(bot, centre, aim, r0, key) {
  const trace = [];
  bot.setControlState(key, true);
  for (let t = 0; t < MOTION_TICKS; t++) {
    const pos = bot.entity.position;
    const vel = bot.entity.velocity;
    const dx = pos.x - centre.x;
    const dz = pos.z - centre.z;
    const r = Math.hypot(dx, dz);
    const vr = r > 1e-6 ? (vel.x * dx + vel.z * dz) / r : 0;
    const rPred = r + vr * RADIUS_PREDICT_TICKS;
    const fwd = rPred > r0 + RADIUS_DEADBAND;
    const back = !fwd && rPred < r0 - RADIUS_DEADBAND;
    bot.setControlState("forward", fwd);
    bot.setControlState("back", back);
    const a = anglesTo(eyeOf(pos), aim);
    await bot.look(a.yaw, a.pitch, true);
    trace.push({
      t,
      time_ms: Date.now(),
      x: +pos.x.toFixed(3),
      y: +pos.y.toFixed(3),
      z: +pos.z.toFixed(3),
      r: +r.toFixed(3),
      bearing_deg: +(bearingOf(centre, pos) / DEG).toFixed(2),
      yaw: +a.yaw.toFixed(4),
      pitch: +a.pitch.toFixed(4),
      forward: fwd,
      back: back,
    });
    await bot.waitForTicks(1);
  }
  stopAll(bot);
  return trace;
}

function traceStats(trace, r0) {
  const rs = trace.map((p) => p.r);
  let stuck = 0;
  for (let i = 10; i < trace.length; i++) {
    const d = Math.hypot(trace[i].x - trace[i - 1].x, trace[i].z - trace[i - 1].z);
    if (d < 0.02) stuck++;
  }
  const bearings = trace.map((p) => p.bearing_deg * DEG);
  let swept = 0;
  for (let i = 1; i < bearings.length; i++) swept += wrapRad(bearings[i] - bearings[i - 1]);
  return {
    r_min: Math.min(...rs),
    r_max: Math.max(...rs),
    r_mean: rs.reduce((s, v) => s + v, 0) / rs.length,
    r_err_max: Math.max(...rs.map((r) => Math.abs(r - r0))),
    r_final_err: rs[rs.length - 1] - r0,
    stuck_ticks: stuck,
    swept_deg: swept / DEG,
    forward_ticks: trace.filter((p) => p.forward).length,
    back_ticks: trace.filter((p) => p.back).length,
    y_min: Math.min(...trace.map((p) => p.y)),
    y_max: Math.max(...trace.map((p) => p.y)),
  };
}

// ---------------------------------------------------------------------------
// Phase
// ---------------------------------------------------------------------------

function getOnCoObservePhaseFn(
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
      "coObservePhase",
      bot.entity.position.clone(),
      episodeNum,
      "coObservePhase beginning",
    );
    const layout = episodeInstance._layout;
    const stopNow = () => {
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
        "coObservePhase end",
      );
    };

    if (!layout) {
      console.log(`[${bot.username}] no layout (scene search failed), stopping`);
      stopNow();
      return;
    }

    const me = layout.bots[bot.username];
    const centre = layout.centre;
    const aim = layout.aim;
    const meta = episodeInstance._episodeMetadata;
    let trace = [];
    try {
      const a0 = anglesTo(eyeOf(bot.entity.position), aim);
      await bot.look(a0.yaw, a0.pitch, true);
      await bot.waitForTicks(LEAD_TICKS);
      // Both bots start strafing right after this exchange.
      await rendezvous(coordinator, "coObserveGo", episodeNum);
      meta.window_start_time_ms = Date.now();
      trace = await circleStrafe(bot, centre, aim, me.radius, me.strafe_key);
      meta.window_end_time_ms = Date.now();
      await bot.waitForTicks(TAIL_TICKS);
    } finally {
      stopAll(bot);
    }
    meta.orbit_trace = trace;
    meta.orbit_stats = trace.length ? traceStats(trace, me.radius) : null;
    const endPos = bot.entity.position;
    meta.end_pos = [endPos.x, endPos.y, endPos.z];
    console.log(
      `[${bot.username}] circle-strafe done: ${JSON.stringify(meta.orbit_stats)}`,
    );
    stopNow();
  };
}

/**
 * Shared-world-consistency eval episode ("co-observe"). A colourful concrete
 * landmark stands on a flat patch; both bots circle-strafe around it on
 * level rings (radius 6-12), looking at it every tick, either in the same
 * angular direction or in opposite directions (alternating by episode). The
 * world does not change while recording. The initial angle between the two
 * lines of sight is 30-90 deg.
 *
 * episode_info (both bots) carries the centre, the landmark columns, the
 * aim point, per-bot radius / start / strafe key, the direction mode, the
 * initial view angle and the motion window (window_start_time_ms /
 * window_end_time_ms -> window_start_frame / window_end_frame in
 * postprocess); each bot adds its own per-tick orbit trace and stats.
 * @extends BaseEpisode
 */
class CoObserveEvalEpisode extends BaseEpisode {
  static WORKS_IN_NON_FLAT_WORLD = true;
  static INIT_MIN_BOTS_DISTANCE = 4;
  static INIT_MAX_BOTS_DISTANCE = 8;

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
    const seed = `coObserveEval_${CO_OBSERVE_BASE_SEED}_${args.instance_id ?? 0}_${episodeNum}`;
    const isPrimary = decidePrimaryBot(bot, sharedBotRng, args);
    this._isPrimary = isPrimary;
    this._layout = null;
    const primaryName = isPrimary ? bot.username : args.other_bot_name;
    this._episodeMetadata = {
      co_observe_seed: seed,
      primary: primaryName,
      role: isPrimary ? "primary" : "secondary",
      lead_ticks: LEAD_TICKS,
      motion_ticks: MOTION_TICKS,
      tail_ticks: TAIL_TICKS,
      scene_check_passed: false,
    };
    this._evalMetadata = {
      bots_chosen: [bot.username, args.other_bot_name].sort(),
      primary: primaryName,
    };

    await quietNotificationsOnce(bot, rcon);

    let layoutPromise = null;
    if (!isPrimary) {
      layoutPromise = new Promise((resolve) => {
        coordinator.onceEvent("coObserveLayout", episodeNum, (data) => resolve(data));
      });
    }
    await rendezvous(coordinator, "coObserveReady", episodeNum);

    let layout = null;
    if (isPrimary) {
      layout = await this.primarySetup(bot, rcon, episodeNum, args, botPosition, seed);
      coordinator.sendToOtherBot("coObserveLayout", { layout }, episodeNum, "coObserveLayout send");
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
      aim_point: [layout.aim.x, layout.aim.y, layout.aim.z],
      landmark: layout.landmark,
      direction_mode: layout.direction_mode,
      initial_view_angle_deg: layout.initial_view_angle_deg,
      bots: layout.bots,
      radius_target: me.radius,
      strafe_key: me.strafe_key,
      start_pos: me.start_pos,
      start_bearing_deg: me.start_bearing_deg,
      arena: layout.arena,
      terraform: layout.terraform,
      scene_search_attempts: layout.attempts,
      bearing_convention: "atan2(z - cz, x - cx); strafe left increases the bearing",
    });
    Object.assign(this._evalMetadata, {
      direction_mode: layout.direction_mode,
      initial_view_angle_deg: layout.initial_view_angle_deg,
      radius_target: me.radius,
    });

    // The caller aims the camera at otherBotPositionNew with a feet-based
    // look; shift the aim point down by the eye height so the resulting
    // yaw/pitch are the eye-based ones.
    const p = me.start_pos;
    return {
      botPositionNew: new Vec3(p[0], p[1], p[2]),
      otherBotPositionNew: new Vec3(layout.aim.x, layout.aim.y - EYE_HEIGHT, layout.aim.z),
    };
  }

  async primarySetup(bot, rcon, episodeNum, args, botPosition, seed) {
    const rng = seedrandom(seed);
    const plan = buildPlan(rng, episodeNum);
    console.log(
      `[${bot.username}] Episode ${episodeNum}: primary, mode=${plan.mode} keys=${plan.keyA}/${plan.keyB} ` +
        `theta0=${plan.theta0.toFixed(1)} deg radii ${plan.rA}/${plan.rB}`,
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
      scene = await pickScene(bot, plan, rng, limits);
      if (scene) scene.terrain.limits = limits;
    }
    this._episodeMetadata.scene_search_attempts = attempts;
    this._episodeMetadata.scene_hops = hops;
    if (scene === null) return null;
    const { centre, arena } = scene;
    console.log(
      `[${bot.username}] scene: centre (${centre.x}, ${centre.y}, ${centre.z}) arena half ${arena.half} ` +
        `terrain ${JSON.stringify(scene.terrain)} ` +
        `rA ${plan.rA} @ ${(scene.bearingA / DEG).toFixed(0)} deg, rB ${plan.rB} @ ${(scene.bearingB / DEG).toFixed(0)} deg ` +
        `after ${scene.candidates} candidates, ${attempts} attempt(s)`,
    );

    const other = args.other_bot_name;
    const bots = {
      [bot.username]: {
        name: bot.username,
        radius: plan.rA,
        strafe_key: plan.keyA,
        start_pos: [scene.startA.x, scene.startA.y, scene.startA.z],
        start_bearing_deg: scene.bearingA / DEG,
      },
      [other]: {
        name: other,
        radius: plan.rB,
        strafe_key: plan.keyB,
        start_pos: [scene.startB.x, scene.startB.y, scene.startB.z],
        start_bearing_deg: scene.bearingB / DEG,
      },
    };

    // No block drops (leaves of trimmed trees decay during the episode and
    // would drop saplings the bots pick up), no mobs or loot.
    for (const cmd of ["gamerule doTileDrops false", "gamerule doMobLoot false", "gamerule doMobSpawning false"]) {
      try {
        const res = await rcon.send(cmd);
        console.log(`[${bot.username}] ${cmd} -> ${String(res).trim()}`);
      } catch (err) {
        console.log(`[${bot.username}] ${cmd} failed: ${err.message}`);
      }
    }

    // Level the arena, then place both bots on it.
    const terra = await terraform(bot, rcon, arena);
    const terraformInfo = { ...terra, terrain_before: scene.terrain };
    terraformInfo.repair_rounds = await repairArena(bot, rcon, arena, terra.top_block);
    await sleep(1000);
    for (const b of Object.values(bots)) {
      const res = await rcon.send(`tp ${b.name} ${b.start_pos[0]} ${b.start_pos[1]} ${b.start_pos[2]}`);
      console.log(`[${bot.username}] tp ${b.name} -> ${String(res).trim()}`);
    }
    await sleep(1500);
    for (let round = 0; round < 2; round++) {
      const bad = Object.values(bots).filter((b) => !isStandable(bot, b.start_pos[0], b.start_pos[1], b.start_pos[2]));
      if (bad.length === 0) break;
      if (round === 1) {
        console.log(`[${bot.username}] start of ${bad.map((b) => b.name).join(",")} not standable after terraform, giving up`);
        return null;
      }
      for (const b of bad) {
        await repairSpot(bot, rcon, { x: b.start_pos[0], z: b.start_pos[2] }, arena, terra.top_block);
      }
      terraformInfo.repaired_starts = bad.map((b) => b.name);
      await sleep(1000);
      for (const b of bad) {
        await rcon.send(`tp ${b.name} ${b.start_pos[0]} ${b.start_pos[1]} ${b.start_pos[2]}`);
      }
      await sleep(1000);
    }
    const mine = bots[bot.username].start_pos;
    if (bot.entity.position.distanceTo(new Vec3(mine[0], mine[1], mine[2])) > 0.3) {
      console.log(`[${bot.username}] primary not at start (${bot.entity.position}), giving up`);
      return null;
    }

    // Remove mobs and dropped items, then build the landmark. All of it
    // before the recording starts.
    const cx = Math.floor(centre.x);
    const cz = Math.floor(centre.z);
    const killCmd = `execute positioned ${cx} ${centre.y} ${cz} run kill @e[type=!minecraft:player,distance=..${MOB_CLEAR_RADIUS}]`;
    try {
      const res = await rcon.send(killCmd);
      console.log(`[${bot.username}] ${killCmd} -> ${String(res).trim()}`);
    } catch (err) {
      console.log(`[${bot.username}] ${killCmd} failed: ${err.message}`);
    }
    const blocks = landmarkBlocks(centre, plan.shape);
    const landmarkOk = await buildLandmark(bot, rcon, blocks);
    this._episodeMetadata.landmark_ok = landmarkOk;
    if (!landmarkOk) return null;
    await sleep(500);
    try {
      await rcon.send(killCmd);
    } catch (err) {
      console.log(`[${bot.username}] second ${killCmd} failed: ${err.message}`);
    }

    const aim = { x: centre.x, y: centre.y + AIM_HEIGHT, z: centre.z };
    const bounds = landmarkBounds(blocks);
    return {
      centre,
      aim,
      landmark: {
        kind: "random_concrete_structure",
        // Bounding box of all blocks (block coordinates, inclusive).
        footprint_min: bounds.min,
        footprint_max: bounds.max,
        height: bounds.max[1] - centre.y + 1,
        column_heights: plan.shape.columns,
        protrusions: plan.shape.protrusions.map((b) => ({ dx: b.dx, dy: b.dy, dz: b.dz })),
        num_blocks: blocks.length,
        blocks,
      },
      direction_mode: plan.mode,
      initial_view_angle_deg: plan.theta0,
      bots,
      arena,
      terraform: terraformInfo,
      attempts,
    };
  }

  async entryPoint(bot, rcon, sharedBotRng, coordinator, iterationID, episodeNum, args) {
    coordinator.onceEvent(
      "coObservePhase",
      episodeNum,
      getOnCoObservePhaseFn(bot, rcon, sharedBotRng, coordinator, episodeNum, this, args),
    );
    coordinator.sendToOtherBot(
      "coObservePhase",
      bot.entity.position.clone(),
      episodeNum,
      "entryPoint end",
    );
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
      const lm = this._layout.landmark;
      try {
        await rcon.send(
          `fill ${lm.footprint_min.join(" ")} ${lm.footprint_max.join(" ")} minecraft:air`,
        );
      } catch (err) {
        console.log(`[${bot.username}] landmark cleanup failed: ${err.message}`);
      }
    }
  }
}

module.exports = { CoObserveEvalEpisode };

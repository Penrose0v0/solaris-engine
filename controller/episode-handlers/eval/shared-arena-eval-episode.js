const seedrandom = require("seedrandom");
const { Vec3 } = require("vec3");

const { stopAll } = require("../../primitives/movement");
const { decidePrimaryBot } = require("../../utils/coordination");
const { sleep } = require("../../utils/helpers");
const { BaseEpisode } = require("../base-episode");
const { sceneUtils } = require("./traj-segments-eval-episode");
const { coObserveUtils } = require("./co-observe-eval-episode");

const { isStandable, blockAtXYZ, rayHitDistance, EYE_HEIGHT } = sceneUtils;
const {
  surfaceMap,
  rateArena,
  terraform,
  repairArena,
  hop,
  quietNotificationsOnce,
  rendezvous,
  eyeOf,
  wrapRad,
  shuffled,
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
const SHARED_ARENA_BASE_SEED = 20261025;

// Scene type per episode: (instance id + episode number) parity, so a
// 2-world x 5-episode trial gets five of each and both worlds host both.
const SCENE_TYPES = ["courtyard", "village"];

// ---------------------------------------------------------------------------
// Courtyard: a square of wall modules around an INNER_SIZE x INNER_SIZE
// floor, no roof. Modules are 3 wide (MODULE_WIDTH), MODULE_HEIGHT_RANGE
// tall, up to 2 thick (outwards), with pillars / eaves allowed to protrude
// one block inwards. The inner wall face is at WALL_OFFSET from the centre
// column; the levelled square reaches ARENA_MARGIN beyond the thickest
// module and is cleared up to the tallest module plus one.
// ---------------------------------------------------------------------------
const INNER_SIZE = 15;
const INNER_HALF = (INNER_SIZE - 1) / 2; // 7
const WALL_OFFSET = INNER_HALF + 1; // 8
const MODULE_WIDTH = 3;
const MODULES_PER_SIDE = INNER_SIZE / MODULE_WIDTH; // 5
const MODULE_HEIGHT_RANGE = [4, 6];
const MAX_THICKNESS = 2;
const ARENA_MARGIN = 1;
const COURTYARD_ARENA_HALF = WALL_OFFSET + MAX_THICKNESS - 1 + ARENA_MARGIN; // 10

// Material palettes: wood + stone pairs. `wall2` is the stone variant,
// `accent` a glazed terracotta. No transparent, falling, liquid or animated
// blocks anywhere (lanterns are static, torches are not used).
const PALETTES = {
  oak_stone: {
    planks: "oak_planks", log: "oak_log", wall: "stone_bricks", wall2: "mossy_stone_bricks", wall3: "chiseled_stone_bricks",
    slab: "stone_brick_slab", stairs: "stone_brick_stairs", door: "oak_door", accent: "light_gray_glazed_terracotta",
    floor: ["cobblestone", "stone_bricks", "mossy_cobblestone", "andesite", "oak_planks"],
  },
  spruce_deepslate: {
    planks: "spruce_planks", log: "spruce_log", wall: "deepslate_bricks", wall2: "deepslate_tiles", wall3: "polished_deepslate",
    slab: "deepslate_brick_slab", stairs: "deepslate_brick_stairs", door: "spruce_door", accent: "cyan_glazed_terracotta",
    floor: ["cobbled_deepslate", "deepslate_bricks", "spruce_planks", "tuff", "polished_deepslate"],
  },
  birch_sandstone: {
    planks: "birch_planks", log: "birch_log", wall: "sandstone", wall2: "cut_sandstone", wall3: "chiseled_sandstone",
    slab: "sandstone_slab", stairs: "sandstone_stairs", door: "birch_door", accent: "yellow_glazed_terracotta",
    floor: ["smooth_sandstone", "sandstone", "birch_planks", "mud_bricks", "packed_mud"],
  },
  cherry_terracotta: {
    planks: "cherry_planks", log: "cherry_log", wall: "bricks", wall2: "terracotta", wall3: "red_terracotta",
    slab: "brick_slab", stairs: "brick_stairs", door: "cherry_door", accent: "pink_glazed_terracotta",
    floor: ["bricks", "terracotta", "cherry_planks", "mud_bricks", "polished_granite"],
  },
};

// A module returns voxels {u, v, w, block} for a palette P and height H:
// u 0..2 along the wall, v 0..H-1 up (v == H allowed for lanterns on the
// top), w 0 = inner layer, 1 = outer layer, -1 = protruding inwards. `ctx`
// carries the side-dependent block states: ctx.inward (facing towards the
// yard), ctx.outward, ctx.axis (log axis along the wall).
function solidBox(P, H, block, w = 0) {
  const out = [];
  for (let u = 0; u < MODULE_WIDTH; u++) for (let v = 0; v < H; v++) out.push({ u, v, w, block });
  return out;
}

const MODULES = {
  // Stone wall with a 1x2 window niche (inner layer open, outer layer solid).
  stone_window: (P, H) => [
    ...solidBox(P, H, P.wall, 0).filter((b) => !(b.u === 1 && (b.v === 1 || b.v === 2))),
    ...solidBox(P, H, P.wall2, 1),
    { u: 1, v: 3, w: 0, block: P.slab + "[type=bottom]" },
  ],
  // Log columns with a planks infill and a log beam on top.
  timber_frame: (P, H, ctx) => {
    const out = [];
    for (let v = 0; v < H; v++) {
      out.push({ u: 0, v, w: 0, block: P.log }, { u: 2, v, w: 0, block: P.log });
      out.push({ u: 1, v, w: 0, block: v === H - 1 ? `${P.log}[axis=${ctx.axis}]` : P.planks });
    }
    return out;
  },
  // Planks frame with rows of bookshelves.
  bookshelf: (P, H) => solidBox(P, H, P.planks).map((b) => (b.v >= 1 && b.v <= H - 2 ? { ...b, block: "bookshelf" } : b)),
  // Planks wall with a door in the middle and lanterns on top of the sides.
  door_lanterns: (P, H, ctx) => [
    ...solidBox(P, H, P.planks).filter((b) => !(b.u === 1 && b.v <= 1)),
    { u: 1, v: 0, w: 0, block: `${P.door}[facing=${ctx.inward},half=lower]` },
    { u: 1, v: 1, w: 0, block: `${P.door}[facing=${ctx.inward},half=upper]` },
    { u: 0, v: H, w: 0, block: "lantern" },
    { u: 2, v: H, w: 0, block: "lantern" },
  ],
  // Fireplace niche: inner layer open at the bottom centre, chimney of wall3 above.
  fireplace_niche: (P, H) => [
    ...solidBox(P, H, P.wall, 0).filter((b) => !(b.u === 1 && b.v <= 1)).map((b) => (b.u === 1 ? { ...b, block: P.wall3 } : b)),
    ...solidBox(P, H, P.wall2, 1),
  ],
  // Stone wall with a glazed terracotta band and single accents.
  glazed_band: (P, H) => solidBox(P, H, P.wall).map((b) => (b.v === 2 || (b.u === 1 && (b.v === 0 || b.v === H - 1)) ? { ...b, block: P.accent } : b)),
  // Stone wall with an eave of upper slabs protruding inwards under the top row.
  eave_slab: (P, H) => [
    ...solidBox(P, H, P.wall2),
    { u: 0, v: H - 1, w: -1, block: P.slab + "[type=top]" },
    { u: 1, v: H - 1, w: -1, block: P.slab + "[type=top]" },
    { u: 2, v: H - 1, w: -1, block: P.slab + "[type=top]" },
  ],
  // Planks wall with a log pillar protruding inwards, capped with a slab.
  pillar_out: (P, H) => [
    ...solidBox(P, H, P.planks),
    ...Array.from({ length: H }, (_, v) => ({ u: 1, v, w: -1, block: P.log })),
    { u: 1, v: H, w: -1, block: P.slab + "[type=bottom]" },
  ],
  // Stone wall mixing the two stone variants (seeded noise, see build).
  brick_mixed: (P, H) => solidBox(P, H, P.wall).map((b) => ({ ...b, mix: true })),
  // Arched alcove: inner layer open 1x3 with a stair arch on top, outer layer solid.
  arched_alcove: (P, H, ctx) => [
    ...solidBox(P, H, P.wall, 0).filter((b) => !(b.u === 1 && b.v <= 2)),
    ...solidBox(P, H, P.wall2, 1),
    { u: 0, v: 3, w: -1, block: `${P.stairs}[facing=${ctx.outward},half=top]` },
    { u: 2, v: 3, w: -1, block: `${P.stairs}[facing=${ctx.outward},half=top]` },
  ],
  // Horizontal logs along the wall alternating with planks rows.
  log_stack: (P, H, ctx) => solidBox(P, H, P.planks).map((b) => (b.v % 2 === 0 ? { ...b, block: `${P.log}[axis=${ctx.axis}]` } : b)),
  // Glazed terracotta centre panel between log columns.
  terracotta_panel: (P, H) => solidBox(P, H, P.log).map((b) => (b.u === 1 ? { ...b, block: b.v === 0 || b.v === H - 1 ? P.wall3 : P.accent } : b)),
  // Rough stone base, planks above, log on top.
  cobble_base: (P, H, ctx) => solidBox(P, H, P.planks).map((b) => (b.v <= 1 ? { ...b, block: P.wall2 } : b.v === H - 1 ? { ...b, block: `${P.log}[axis=${ctx.axis}]` } : b)),
  // Buttressed stone: wall3 columns at the sides protruding inwards on the lower half.
  buttress: (P, H) => [
    ...solidBox(P, H, P.wall),
    ...[0, 2].flatMap((u) => Array.from({ length: Math.ceil(H / 2) }, (_, v) => ({ u, v, w: -1, block: P.wall3 }))),
  ],
};

// Side geometry: along-wall axis, outward normal and block-state names.
const SIDES = {
  south: { axis: "x", inward: "north", outward: "south", place: (cx, cz, along, w) => [cx - INNER_HALF + along, cz + WALL_OFFSET + w] },
  north: { axis: "x", inward: "south", outward: "north", place: (cx, cz, along, w) => [cx - INNER_HALF + along, cz - WALL_OFFSET - w] },
  east: { axis: "z", inward: "west", outward: "east", place: (cx, cz, along, w) => [cx + WALL_OFFSET + w, cz - INNER_HALF + along] },
  west: { axis: "z", inward: "east", outward: "west", place: (cx, cz, along, w) => [cx - WALL_OFFSET - w, cz - INNER_HALF + along] },
};

/**
 * Courtyard shape relative to the centre column: floor at dy -1, modules
 * around the four sides, log pillars on the corners. Returns the block
 * list (dx, dy, dz, block with state, module, part) and the layout.
 */
function buildCourtyardShape(rng) {
  const paletteName = Object.keys(PALETTES)[Math.floor(rng() * Object.keys(PALETTES).length)];
  const P = PALETTES[paletteName];
  const names = Object.keys(MODULES);
  const blocks = [];
  const layout = [];
  const seen = new Set();
  const push = (dx, dy, dz, block, module, part) => {
    const k = `${dx},${dy},${dz}`;
    if (seen.has(k)) return;
    seen.add(k);
    blocks.push({ dx, dy, dz, block, module, part });
  };
  // Floor.
  for (let dx = -INNER_HALF; dx <= INNER_HALF; dx++) {
    for (let dz = -INNER_HALF; dz <= INNER_HALF; dz++) {
      push(dx, -1, dz, P.floor[Math.floor(rng() * P.floor.length)], "floor", "floor");
    }
  }
  let maxH = 0;
  for (const [side, S] of Object.entries(SIDES)) {
    const order = shuffled(names, rng).slice(0, MODULES_PER_SIDE);
    for (let k = 0; k < MODULES_PER_SIDE; k++) {
      const name = order[k];
      const H = MODULE_HEIGHT_RANGE[0] + Math.floor(rng() * (MODULE_HEIGHT_RANGE[1] - MODULE_HEIGHT_RANGE[0] + 1));
      maxH = Math.max(maxH, H);
      const ctx = { inward: S.inward, outward: S.outward, axis: S.axis };
      const voxels = MODULES[name](P, H, ctx);
      layout.push({ side, index: k, module: name, height: H });
      for (const vx of voxels) {
        const block = vx.mix ? (rng() < 0.3 ? P.wall2 : rng() < 0.15 ? P.wall3 : P.wall) : vx.block;
        const [dx, dz] = S.place(0, 0, k * MODULE_WIDTH + vx.u, vx.w);
        push(dx, vx.v, dz, block, name, vx.w < 0 ? "protrusion" : "wall");
      }
    }
  }
  // Corner pillars (logs) up to the tallest module, lantern on top.
  for (const sx of [-1, 1]) {
    for (const sz of [-1, 1]) {
      for (let v = 0; v < maxH; v++) push(sx * WALL_OFFSET, v, sz * WALL_OFFSET, P.log, "corner", "wall");
      push(sx * WALL_OFFSET, maxH, sz * WALL_OFFSET, "lantern", "corner", "wall");
    }
  }
  return { palette: paletteName, layout, blocks, max_height: maxH + 1 };
}

// ---------------------------------------------------------------------------
// Looks and standing constraints (mineflayer convention: yaw 0 = north = -z,
// yaw grows counter-clockwise, pitch positive = up)
// ---------------------------------------------------------------------------
const WALL_CLEARANCE = 3;
const BOT_DISTANCE = [4, 8];
const OVERLAP_SHIFT_BLOCKS = 2.0;
const OVERLAP_YAW_JITTER_DEG = 15;
const INITIAL_PITCH_RANGE_DEG = [-5, 5];
// |baseline . forward| at frame 0 for each bot: no standing one behind the
// other (COLMAP cannot initialise from a near-zero-parallax pair).
const MAX_BASELINE_FORWARD_DOT = 0.7;
const PLACEMENT_TRIES = 400;

const LEAD_TICKS = 10;
const MOTION_TICKS = 160;
const TAIL_TICKS = 25;
const SWEEP_RANGE_DEG = [90, 270];
const OMEGA_RANGE_DEG_PER_SEC = [30, 90];
const OMEGA_QUANTUM_DEG_PER_SEC = 3;
const PITCH_AMP_RANGE_DEG = [0, 10];
const PITCH_PERIOD_RANGE_S = [3, 6];
const PITCH_LIMIT_DEG = 60;

function uniform(rng, lo, hi) {
  return lo + rng() * (hi - lo);
}

function viewDir(yaw) {
  return { x: -Math.sin(yaw), z: -Math.cos(yaw) };
}

function yawTo(from, to) {
  return Math.atan2(-(to.x - from.x), -(to.z - from.z));
}

// |baseline . forward| for both bots (horizontal unit vectors).
function baselineDots(posA, posB, yawA, yawB) {
  const bx = posB.x - posA.x;
  const bz = posB.z - posA.z;
  const n = Math.hypot(bx, bz) || 1;
  const dA = viewDir(yawA);
  const dB = viewDir(yawB);
  return { a: Math.abs((bx * dA.x + bz * dA.z) / n), b: Math.abs((bx * dB.x + bz * dB.z) / n) };
}

function sweepPlan(rng) {
  const kMin = Math.ceil(OMEGA_RANGE_DEG_PER_SEC[0] / OMEGA_QUANTUM_DEG_PER_SEC);
  const kMax = Math.floor(OMEGA_RANGE_DEG_PER_SEC[1] / OMEGA_QUANTUM_DEG_PER_SEC);
  return {
    sweep_deg: uniform(rng, SWEEP_RANGE_DEG[0], SWEEP_RANGE_DEG[1]),
    omega_deg_per_sec: OMEGA_QUANTUM_DEG_PER_SEC * Math.floor(uniform(rng, kMin, kMax + 1)),
    direction: rng() < 0.5 ? 1 : -1,
    pitch_amp_deg: uniform(rng, PITCH_AMP_RANGE_DEG[0], PITCH_AMP_RANGE_DEG[1]),
    pitch_period_s: uniform(rng, PITCH_PERIOD_RANGE_S[0], PITCH_PERIOD_RANGE_S[1]),
    pitch_phase: rng() * 2 * Math.PI,
    pitch0: uniform(rng, INITIAL_PITCH_RANGE_DEG[0], INITIAL_PITCH_RANGE_DEG[1]) * DEG,
  };
}

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

/**
 * Point where a horizontal ray from `pos` along `yaw` meets the inner wall
 * faces of the courtyard (at centre +- (WALL_OFFSET - 0.5)).
 */
function wallHit(centre, pos, yaw) {
  const d = viewDir(yaw);
  const lim = WALL_OFFSET - 0.5;
  let best = null;
  for (const p of [
    { axis: "x", value: centre.x + lim, wall: "east" },
    { axis: "x", value: centre.x - lim, wall: "west" },
    { axis: "z", value: centre.z + lim, wall: "south" },
    { axis: "z", value: centre.z - lim, wall: "north" },
  ]) {
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

function wallsAdjacent(a, b) {
  const idx = { east: 0, south: 1, west: 2, north: 3 };
  return a === b || (idx[a] - idx[b] + 4) % 2 === 1;
}

/**
 * Courtyard placement: two cells at least WALL_CLEARANCE from the walls,
 * BOT_DISTANCE apart; A looks in a random direction, B at A's wall point
 * (shifted, jittered); both |baseline . forward| <= MAX_BASELINE_FORWARD_DOT.
 */
function planCourtyardPlacement(rng, centre) {
  const lim = Math.floor(WALL_OFFSET - 0.5 - WALL_CLEARANCE); // 4
  const cell = () => ({ dx: Math.floor(uniform(rng, -lim, lim + 1)), dz: Math.floor(uniform(rng, -lim, lim + 1)) });
  for (let tries = 0; tries < PLACEMENT_TRIES; tries++) {
    const a = cell();
    const b = cell();
    const d = Math.hypot(a.dx - b.dx, a.dz - b.dz);
    if (d < BOT_DISTANCE[0] || d > BOT_DISTANCE[1]) continue;
    const posA = { x: centre.x + a.dx, y: centre.y, z: centre.z + a.dz };
    const posB = { x: centre.x + b.dx, y: centre.y, z: centre.z + b.dz };
    const yawA = uniform(rng, -Math.PI, Math.PI);
    const hitA = wallHit(centre, eyeOf(posA), yawA);
    if (!hitA) continue;
    const alongX = hitA.wall === "north" || hitA.wall === "south";
    const shift = uniform(rng, -OVERLAP_SHIFT_BLOCKS, OVERLAP_SHIFT_BLOCKS);
    const target = { x: hitA.x + (alongX ? shift : 0), z: hitA.z + (alongX ? 0 : shift) };
    const yawB = wrapRad(yawTo(eyeOf(posB), target) + uniform(rng, -OVERLAP_YAW_JITTER_DEG, OVERLAP_YAW_JITTER_DEG) * DEG);
    const hitB = wallHit(centre, eyeOf(posB), yawB);
    if (!hitB || !wallsAdjacent(hitA.wall, hitB.wall)) continue;
    const dots = baselineDots(posA, posB, yawA, yawB);
    if (dots.a > MAX_BASELINE_FORWARD_DOT || dots.b > MAX_BASELINE_FORWARD_DOT) continue;
    return { posA, posB, yawA, yawB, hitA, hitB, dots, tries: tries + 1 };
  }
  return null;
}

// ---------------------------------------------------------------------------
// Village: standing spots found by ray coverage, no block changes.
// ---------------------------------------------------------------------------
const VILLAGE_LOCATE = "#minecraft:village";
const VILLAGE_LAND_RADIUS = 8;
const VILLAGE_SPOT_RADIUS = 24;
const VILLAGE_SPOT_STRIDE = 2;
const VILLAGE_MAX_ATTEMPTS = 12;
const VILLAGE_SETTLE_MS = 6000;
const VILLAGE_HOP_RANGE = 600;
const RAY_STEP_DEG = 15;
const RAY_PITCHES_DEG = [0, 8, 16];
const RAY_MAX_DIST = 12;
const RAY_MIN_COVERAGE = 0.8;
// Ray coverage alone prefers caves, overhangs and cliff pockets (trial002
// run 1). A village spot must also be outdoors (sky light at the eye) and
// most of what the rays hit must be built material, not terrain.
const MIN_SKY_LIGHT = 14;
const MIN_BUILT_FRACTION = 0.4;
// Nothing solid straight above the eye for this many blocks: not under a
// roof or awning (trial002 run 2 still put bots under house eaves).
const OPEN_ABOVE_BLOCKS = 10;
const BUILT_BLOCK_RE =
  /(planks|cobblestone|stripped_|_door|_fence|_stairs|_slab|bell|hay_block|lantern|torch|dirt_path|smooth_sandstone|cut_sandstone|chiseled_sandstone|terracotta|bricks|glass|campfire|barrel|composter|crafting_table|lectern|smithing_table|loom|cartography_table|fletching_table|furnace|smoker|grindstone|stonecutter|cauldron|_bed|_wool|bookshelf|mossy|_trapdoor|_wall$|chest|jack_o_lantern|carved_pumpkin|melon|cobweb)/;
const LOOK_HIT_RANGE = [3, 12];
const VILLAGE_EXPORT_RADIUS = 16;
const VILLAGE_EXPORT_DY = [-4, 16];
const VILLAGE_SPOT_BUDGET_MS = 25000;

// First full block along a ray (step 0.5): {t, name} or null.
function rayHitInfo(bot, eye, yaw, pitch, maxDist) {
  const ch = Math.cos(pitch);
  const dx = -Math.sin(yaw) * ch;
  const dy = Math.sin(pitch);
  const dz = -Math.cos(yaw) * ch;
  for (let t = 0.5; t <= maxDist; t += 0.5) {
    const b = blockAtXYZ(bot, eye.x + dx * t, eye.y + dy * t, eye.z + dz * t);
    if (b === null) return null;
    if (b.boundingBox === "block") return { t, name: b.name };
  }
  return null;
}

// Fraction of rays (every RAY_STEP_DEG horizontally, at RAY_PITCHES_DEG)
// from the eye at `pos` that hit a full block within RAY_MAX_DIST, the
// fraction of those hits on built (village) material, and the sky light.
function rayCoverage(bot, pos) {
  const eye = eyeOf(pos);
  let hits = 0;
  let built = 0;
  let total = 0;
  const perYaw = [];
  for (let yawDeg = 0; yawDeg < 360; yawDeg += RAY_STEP_DEG) {
    let h = 0;
    for (const pitchDeg of RAY_PITCHES_DEG) {
      total++;
      const hit = rayHitInfo(bot, eye, yawDeg * DEG, pitchDeg * DEG, RAY_MAX_DIST);
      if (hit) {
        hits++;
        h++;
        if (BUILT_BLOCK_RE.test(hit.name)) built++;
      }
    }
    perYaw.push(h);
  }
  const eyeBlock = bot.blockAt(new Vec3(Math.floor(eye.x), Math.floor(eye.y), Math.floor(eye.z)));
  const skyLight = eyeBlock && typeof eyeBlock.skyLight === "number" ? eyeBlock.skyLight : null;
  let openAbove = true;
  for (let dy = 1; dy <= OPEN_ABOVE_BLOCKS; dy++) {
    const b = blockAtXYZ(bot, eye.x, eye.y + dy, eye.z);
    if (b !== null && b.boundingBox !== "empty") {
      openAbove = false;
      break;
    }
  }
  return { fraction: hits / total, built_fraction: hits ? built / hits : 0, sky_light: skyLight, open_above: openAbove, per_yaw_hits: perYaw };
}

function spotQualifies(cov) {
  return cov.fraction >= RAY_MIN_COVERAGE && cov.built_fraction >= MIN_BUILT_FRACTION && cov.open_above &&
    (cov.sky_light === null || cov.sky_light >= MIN_SKY_LIGHT);
}

// First full block along the horizontal ray, up to maxDist (step 0.25).
function rayHitPoint(bot, eye, yaw, maxDist) {
  const d = viewDir(yaw);
  for (let t = 0.25; t <= maxDist; t += 0.25) {
    const b = blockAtXYZ(bot, eye.x + d.x * t, eye.y, eye.z + d.z * t);
    if (b === null) return null;
    if (b.boundingBox === "block") return { x: eye.x + d.x * t, y: eye.y, z: eye.z + d.z * t, t };
  }
  return null;
}

// Segment from `from` to `to` free of full blocks (stops `margin` short).
function segmentClear(bot, from, to, margin) {
  const dx = to.x - from.x;
  const dy = to.y - from.y;
  const dz = to.z - from.z;
  const len = Math.hypot(dx, dy, dz);
  for (let t = 0.25; t < len - margin; t += 0.25) {
    const b = blockAtXYZ(bot, from.x + (dx * t) / len, from.y + (dy * t) / len, from.z + (dz * t) / len);
    if (b === null || b.boundingBox === "block") return false;
  }
  return true;
}

/**
 * Village placement around `origin`: standable spots with ray coverage >=
 * RAY_MIN_COVERAGE, two of them BOT_DISTANCE apart; A looks at a building
 * (LOOK_HIT_RANGE), B at A's hit point with jitter and a clear line of
 * sight; |baseline . forward| <= MAX_BASELINE_FORWARD_DOT for both.
 */
async function planVillagePlacement(bot, rng, origin) {
  const base = { x: Math.floor(origin.x), y: Math.floor(origin.y), z: Math.floor(origin.z) };
  const deadline = Date.now() + VILLAGE_SPOT_BUDGET_MS;
  const spots = [];
  let checked = 0;
  let unloaded = 0;
  const rejected = { coverage: 0, built: 0, roof: 0, dark: 0 };
  for (let dx = -VILLAGE_SPOT_RADIUS; dx <= VILLAGE_SPOT_RADIUS; dx += VILLAGE_SPOT_STRIDE) {
    for (let dz = -VILLAGE_SPOT_RADIUS; dz <= VILLAGE_SPOT_RADIUS; dz += VILLAGE_SPOT_STRIDE) {
      if (Date.now() > deadline) break;
      const x = base.x + dx + 0.5;
      const z = base.z + dz + 0.5;
      if (blockAtXYZ(bot, x, base.y, z) === null) {
        unloaded++;
        continue;
      }
      // Highest standable level first: the surface, not a cave below it.
      for (let dy = 3; dy >= -3; dy--) {
        const y = base.y + dy;
        if (!isStandable(bot, x, y, z)) continue;
        checked++;
        const cov = rayCoverage(bot, { x, y, z });
        if (spotQualifies(cov)) {
          spots.push({ pos: { x, y, z }, coverage: cov.fraction, built_fraction: cov.built_fraction, sky_light: cov.sky_light, per_yaw: cov.per_yaw_hits });
        } else if (cov.fraction < RAY_MIN_COVERAGE) {
          rejected.coverage++;
        } else if (cov.built_fraction < MIN_BUILT_FRACTION) {
          rejected.built++;
        } else if (!cov.open_above) {
          rejected.roof++;
        } else {
          rejected.dark++;
        }
        break;
      }
    }
    await sleep(0);
  }
  console.log(
    `[${bot.username}] village spots: ${spots.length} of ${checked} standable columns qualify (rejected ${JSON.stringify(rejected)}, unloaded ${unloaded})`,
  );
  if (spots.length < 2) return { placement: null, spots_found: spots.length, columns_checked: checked, rejected };
  // Most built-up spots first (seeded tie-break).
  const shuffledSpots = shuffled(spots, rng).sort((p, q) => q.built_fraction - p.built_fraction);
  for (let tries = 0; tries < PLACEMENT_TRIES; tries++) {
    const A = shuffledSpots[tries % shuffledSpots.length];
    const partners = spots.filter((s) => {
      const d = Math.hypot(s.pos.x - A.pos.x, s.pos.z - A.pos.z);
      return d >= BOT_DISTANCE[0] && d <= BOT_DISTANCE[1] && Math.abs(s.pos.y - A.pos.y) <= 1;
    });
    if (partners.length === 0) continue;
    const B = partners[Math.floor(rng() * partners.length)];
    const eyeA = eyeOf(A.pos);
    const eyeB = eyeOf(B.pos);
    for (let k = 0; k < 12; k++) {
      const yawA = uniform(rng, -Math.PI, Math.PI);
      const hitA = rayHitPoint(bot, eyeA, yawA, LOOK_HIT_RANGE[1]);
      if (!hitA || hitA.t < LOOK_HIT_RANGE[0]) continue;
      if (!segmentClear(bot, eyeB, hitA, 0.6)) continue;
      const yawB = wrapRad(yawTo(eyeB, hitA) + uniform(rng, -OVERLAP_YAW_JITTER_DEG, OVERLAP_YAW_JITTER_DEG) * DEG);
      const hitB = rayHitPoint(bot, eyeB, yawB, LOOK_HIT_RANGE[1] + 4);
      if (!hitB) continue;
      const dots = baselineDots(A.pos, B.pos, yawA, yawB);
      if (dots.a > MAX_BASELINE_FORWARD_DOT || dots.b > MAX_BASELINE_FORWARD_DOT) continue;
      return {
        placement: {
          posA: A.pos, posB: B.pos, yawA, yawB,
          hitA: { x: hitA.x, y: hitA.y, z: hitA.z, dist: hitA.t }, hitB: { x: hitB.x, y: hitB.y, z: hitB.z, dist: hitB.t },
          dots, coverageA: A.coverage, coverageB: B.coverage, builtA: A.built_fraction, builtB: B.built_fraction,
          skyA: A.sky_light, skyB: B.sky_light, per_yaw_a: A.per_yaw, per_yaw_b: B.per_yaw, tries: tries + 1,
        },
        spots_found: spots.length,
        columns_checked: checked,
        rejected,
      };
    }
  }
  return { placement: null, spots_found: spots.length, columns_checked: checked, rejected };
}

// Non-air blocks within VILLAGE_EXPORT_RADIUS (horizontal) of either bot,
// VILLAGE_EXPORT_DY around the lower bot, from this client's block data.
function exportBlocks(bot, positions) {
  const y0 = Math.min(...positions.map((p) => p.y));
  const out = [];
  const cols = new Set();
  for (const p of positions) {
    const cx = Math.floor(p.x);
    const cz = Math.floor(p.z);
    for (let dx = -VILLAGE_EXPORT_RADIUS; dx <= VILLAGE_EXPORT_RADIUS; dx++) {
      for (let dz = -VILLAGE_EXPORT_RADIUS; dz <= VILLAGE_EXPORT_RADIUS; dz++) {
        if (dx * dx + dz * dz > VILLAGE_EXPORT_RADIUS * VILLAGE_EXPORT_RADIUS) continue;
        cols.add(`${cx + dx},${cz + dz}`);
      }
    }
  }
  let unloaded = 0;
  for (const key of cols) {
    const [x, z] = key.split(",").map(Number);
    for (let y = y0 + VILLAGE_EXPORT_DY[0]; y <= y0 + VILLAGE_EXPORT_DY[1]; y++) {
      const b = bot.blockAt(new Vec3(x, y, z));
      if (b === null) {
        unloaded++;
        continue;
      }
      if (b.name === "air") continue;
      let props = null;
      try {
        props = b.getProperties();
      } catch (err) {
        props = null;
      }
      const entry = [x, y, z, b.name];
      if (props && Object.keys(props).length) entry.push(props);
      out.push(entry);
    }
  }
  return { blocks: out, columns: cols.size, unloaded, y_range: [y0 + VILLAGE_EXPORT_DY[0], y0 + VILLAGE_EXPORT_DY[1]] };
}

async function locateVillage(bot, rcon, x, z) {
  const res = String(await rcon.send(`execute positioned ${x} 64 ${z} run locate structure ${VILLAGE_LOCATE}`));
  const m = /\[(-?\d+), (?:~|-?\d+), (-?\d+)\]/.exec(res);
  console.log(`[${bot.username}] locate village from ${x},${z}: ${res.trim()}`);
  return m ? { x: parseInt(m[1], 10), z: parseInt(m[2], 10), raw: res.trim() } : null;
}

// ---------------------------------------------------------------------------
// Courtyard scene search (level square, see coObserveUtils.rateArena)
// ---------------------------------------------------------------------------
const SCENE_SEARCH_BUDGET_MS = 6000;

async function pickCourtyardArena(bot, limits) {
  const base = bot.entity.position.floored();
  let surface = null;
  for (let i = 0; i <= HEIGHT_MAP_RETRIES; i++) {
    surface = surfaceMap(bot, base, SCENE_SEARCH_RADIUS + COURTYARD_ARENA_HALF);
    if (surface.unloaded / surface.total <= MAX_UNLOADED_FRACTION) break;
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
      const r = rateArena(surface, dx, dz, e.h, COURTYARD_ARENA_HALF, limits);
      if (r.reason) {
        rejected[r.reason]++;
        continue;
      }
      rated.push({ dx, dz, g: e.h, stats: r.stats, d2: dx * dx + dz * dz });
    }
    await sleep(0);
  }
  if (rated.length === 0) {
    console.log(`[${bot.username}] arena search: nothing level enough, rejected by ${JSON.stringify(rejected)}, unloaded ${surface.unloaded}/${surface.total}`);
    return null;
  }
  rated.sort((a, b) => b.stats.level_fraction - a.stats.level_fraction || a.stats.raised_fraction - b.stats.raised_fraction || a.d2 - b.d2);
  const best = rated[0];
  const cx = base.x + best.dx;
  const cz = base.z + best.dz;
  const y = best.g + 1;
  return { centre: { x: cx + 0.5, y, z: cz + 0.5 }, arena: { cx, cz, y, half: COURTYARD_ARENA_HALF }, terrain: best.stats };
}

// ---------------------------------------------------------------------------
// RCON helpers
// ---------------------------------------------------------------------------
const BUILD_VERIFY_TIMEOUT_MS = 10000;
const BUILD_SERVER_SAMPLE = 40;

function baseName(block) {
  return block.split("[")[0];
}

async function buildBlocks(bot, rcon, blocks, rng) {
  let failed = 0;
  for (const b of blocks) {
    const res = String(await rcon.send(`setblock ${b.x} ${b.y} ${b.z} minecraft:${b.block}`));
    if (!/Changed/i.test(res)) {
      failed++;
      if (failed <= 5) console.log(`[${bot.username}] setblock ${b.x} ${b.y} ${b.z} ${b.block} -> ${res.trim()}`);
    }
  }
  let serverBad = 0;
  for (let i = 0; i < BUILD_SERVER_SAMPLE; i++) {
    const b = blocks[Math.floor(rng() * blocks.length)];
    const res = String(await rcon.send(`execute if block ${b.x} ${b.y} ${b.z} minecraft:${baseName(b.block)}`));
    if (!/passed/i.test(res)) serverBad++;
  }
  const deadline = Date.now() + BUILD_VERIFY_TIMEOUT_MS;
  let missing = blocks.length;
  while (missing > 0 && Date.now() < deadline) {
    missing = blocks.filter((b) => {
      const c = bot.blockAt(new Vec3(b.x, b.y, b.z));
      return !(c && c.name === baseName(b.block));
    }).length;
    if (missing > 0) await sleep(250);
  }
  console.log(`[${bot.username}] built ${blocks.length} blocks: setblock not-changed ${failed}, server sample bad ${serverBad}/${BUILD_SERVER_SAMPLE}, client missing ${missing}`);
  return { ok: serverBad === 0 && missing === 0, setblock_not_changed: failed, server_sample_bad: serverBad, client_missing: missing };
}

async function rconTry(bot, rcon, cmd) {
  try {
    const res = String(await rcon.send(cmd)).trim();
    console.log(`[${bot.username}] ${cmd} -> ${res}`);
    return res;
  } catch (err) {
    console.log(`[${bot.username}] ${cmd} failed: ${err.message}`);
    return null;
  }
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
      console.log(`[${bot.username}] no layout (scene setup failed), stopping`);
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
 * Shared-arena eval episode, version 2. Two scene types, alternating with
 * (instance id + episode number):
 *
 *  courtyard  a square courtyard (15x15 inside) walled with seeded
 *             player-style modules (stone with window niches, timber
 *             frames, bookshelves, doors with lanterns, fireplace niches,
 *             glazed bands, eaves, pillars, ...) in one of four material
 *             palettes, textured floor, no roof; built on a levelled spot.
 *  village    a generated village (`locate structure #minecraft:village`),
 *             untouched; standing spots are chosen where >= 80% of the
 *             rays every 15 deg (at 0/8/16 deg pitch) hit a block within
 *             12 blocks; all entities around are removed.
 *
 * In both, the bots stand 4-8 blocks apart, side by side or diagonal
 * (|baseline . forward| <= 0.7 for each at frame 0), start with overlapping
 * views and sweep their yaw in place (triangle wave + small pitch sine).
 * Nothing in the world changes inside the recorded window.
 * @extends BaseEpisode
 */
class SharedArenaEvalEpisode extends BaseEpisode {
  static WORKS_IN_NON_FLAT_WORLD = true;
  static INIT_MIN_BOTS_DISTANCE = 4;
  static INIT_MAX_BOTS_DISTANCE = 8;

  async setupEpisode(bot, rcon, sharedBotRng, coordinator, episodeNum, args, botPosition, otherBotPosition) {
    const seed = `sharedArenaEval_${SHARED_ARENA_BASE_SEED}_${args.instance_id ?? 0}_${episodeNum}`;
    const sceneType = SCENE_TYPES[((args.instance_id ?? 0) + episodeNum) % SCENE_TYPES.length];
    const isPrimary = decidePrimaryBot(bot, sharedBotRng, args);
    this._isPrimary = isPrimary;
    this._layout = null;
    const primaryName = isPrimary ? bot.username : args.other_bot_name;
    this._episodeMetadata = {
      shared_arena_seed: seed,
      scene_type: sceneType,
      primary: primaryName,
      role: isPrimary ? "primary" : "secondary",
      lead_ticks: LEAD_TICKS,
      motion_ticks: MOTION_TICKS,
      tail_ticks: TAIL_TICKS,
      scene_check_passed: false,
    };
    this._evalMetadata = { bots_chosen: [bot.username, args.other_bot_name].sort(), primary: primaryName, scene_type: sceneType };

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
      layout = await this.primarySetup(bot, rcon, episodeNum, args, botPosition, seed, sceneType);
      coordinator.sendToOtherBot("sharedArenaLayout", { layout }, episodeNum, "sharedArenaLayout send");
    } else {
      layout = (await layoutPromise).layout;
      if (layout) await sleep(2500);
    }
    if (!layout) {
      return { botPositionNew: botPosition, otherBotPositionNew: otherBotPosition };
    }
    this._layout = layout;
    const me = layout.bots[bot.username];
    Object.assign(this._episodeMetadata, {
      scene_check_passed: true,
      centre: layout.centre ? [layout.centre.x, layout.centre.y, layout.centre.z] : null,
      courtyard: layout.courtyard || null,
      village: layout.village || null,
      bots: layout.bots,
      start_overlap: layout.start_overlap,
      baseline_forward_dot_planned: layout.start_overlap.baseline_forward_dot,
      pos: me.pos,
      look_plan: { yaw0: me.yaw0, pitch0: me.pitch0, sweep_deg: me.sweep_deg, omega_deg_per_sec: me.omega_deg_per_sec,
        direction: me.direction, pitch_amp_deg: me.pitch_amp_deg, pitch_period_s: me.pitch_period_s, pitch_phase: me.pitch_phase },
      suggested_window: "window_start_frame .. window_end_frame (the 160-tick sweep); frames before it are the static lead",
      arena: layout.arena || null,
      terraform: layout.terraform || null,
      scene_search_attempts: layout.attempts,
      scene_hops: layout.hops,
      world_edits_for_placement: layout.world_edits_for_placement || [],
    });
    Object.assign(this._evalMetadata, { sweep_deg: me.sweep_deg, omega_deg_per_sec: me.omega_deg_per_sec });

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

  async primarySetup(bot, rcon, episodeNum, args, botPosition, seed, sceneType) {
    const rng = seedrandom(seed);
    console.log(`[${bot.username}] Episode ${episodeNum}: primary, scene ${sceneType}`);
    // Freeze random ticks (crop growth, grass spread, leaf decay) and the
    // weather for the episode; mobs and drops off.
    // `weather clear <n>` takes ticks when unitless (1000 = 50 s, and the
    // rain came back mid-episode in trial002 run 2), so stop the weather
    // cycle for the episode and clear with an explicit duration.
    for (const cmd of ["gamerule randomTickSpeed 0", "gamerule doWeatherCycle false", "weather clear 1200s", "gamerule doTileDrops false", "gamerule doMobLoot false", "gamerule doMobSpawning false"]) {
      await rconTry(bot, rcon, cmd);
    }
    return sceneType === "village"
      ? this.villageSetup(bot, rcon, episodeNum, args, botPosition, rng)
      : this.courtyardSetup(bot, rcon, episodeNum, args, botPosition, rng);
  }

  // ---- courtyard -----------------------------------------------------------
  async courtyardSetup(bot, rcon, episodeNum, args, botPosition, rng) {
    const shape = buildCourtyardShape(rng);
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
      scene = await pickCourtyardArena(bot, limits);
      if (scene) scene.terrain.limits = limits;
    }
    this._episodeMetadata.scene_search_attempts = attempts;
    this._episodeMetadata.scene_hops = hops;
    if (scene === null) return null;
    const { centre, arena } = scene;
    console.log(`[${bot.username}] courtyard arena: centre (${centre.x}, ${centre.y}, ${centre.z}) palette ${shape.palette} after ${attempts} attempt(s)`);

    const clearHeight = shape.max_height + 1;
    const terra = await terraform(bot, rcon, arena, clearHeight);
    const terraformInfo = { ...terra, terrain_before: scene.terrain };
    terraformInfo.repair_rounds = await repairArena(bot, rcon, arena, terra.top_block, clearHeight);
    await sleep(500);

    const placement = planCourtyardPlacement(rng, centre);
    if (!placement) {
      console.log(`[${bot.username}] no courtyard placement satisfies the constraints, giving up`);
      return null;
    }
    const sweeps = [sweepPlan(rng), sweepPlan(rng)];
    const names = [bot.username, args.other_bot_name];
    const bots = {};
    [[names[0], placement.posA, placement.yawA, sweeps[0], placement.hitA],
      [names[1], placement.posB, placement.yawB, sweeps[1], placement.hitB]].forEach(([name, pos, yaw0, sw, hit]) => {
      bots[name] = { name, pos: [pos.x, pos.y, pos.z], yaw0, ...sw, wall_hit: hit ? { x: hit.x, z: hit.z, wall: hit.wall } : null };
    });
    const dA = viewDir(placement.yawA);
    const dB = viewDir(placement.yawB);
    const startOverlap = {
      view_angle_deg: Math.acos(Math.max(-1, Math.min(1, dA.x * dB.x + dA.z * dB.z))) / DEG,
      wall_a: placement.hitA.wall,
      wall_b: placement.hitB.wall,
      same_or_adjacent_wall: true,
      bot_distance: Math.hypot(placement.posA.x - placement.posB.x, placement.posA.z - placement.posB.z),
      baseline_forward_dot: placement.dots,
      placement_tries: placement.tries,
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
    await rconTry(bot, rcon, killCmd);
    const blocks = shape.blocks.map((b) => ({ x: cx + b.dx, y: centre.y + b.dy, z: cz + b.dz, block: b.block, module: b.module, part: b.part }));
    const build = await buildBlocks(bot, rcon, blocks, rng);
    this._episodeMetadata.build = build;
    if (!build.ok) return null;
    await rconTry(bot, rcon, killCmd);
    await sleep(500);

    return {
      scene_type: "courtyard",
      centre,
      arena,
      terraform: terraformInfo,
      courtyard: {
        kind: "courtyard_modules",
        inner_size: INNER_SIZE,
        wall_offset: WALL_OFFSET,
        palette: shape.palette,
        palette_blocks: PALETTES[shape.palette],
        modules: shape.layout,
        max_height: shape.max_height,
        box_min: [cx - WALL_OFFSET - MAX_THICKNESS + 1, centre.y - 1, cz - WALL_OFFSET - MAX_THICKNESS + 1],
        box_max: [cx + WALL_OFFSET + MAX_THICKNESS - 1, centre.y + shape.max_height, cz + WALL_OFFSET + MAX_THICKNESS - 1],
        num_blocks: blocks.length,
        blocks,
      },
      bots,
      start_overlap: startOverlap,
      attempts,
      hops,
      world_edits_for_placement: [],
    };
  }

  // ---- village -------------------------------------------------------------
  async villageSetup(bot, rcon, episodeNum, args, botPosition, rng) {
    let placed = null;
    let attempts = 0;
    const hops = [];
    let located = null;
    let searchInfo = null;
    while (placed === null && attempts < VILLAGE_MAX_ATTEMPTS) {
      attempts++;
      const range = attempts === 1 ? 0 : VILLAGE_HOP_RANGE * attempts;
      const ang = rng() * 2 * Math.PI;
      const qx = Math.floor(botPosition.x + range * Math.cos(ang));
      const qz = Math.floor(botPosition.z + range * Math.sin(ang));
      located = await locateVillage(bot, rcon, qx, qz);
      if (!located) {
        hops.push(`attempt ${attempts}: no village located from ${qx},${qz}`);
        continue;
      }
      const res = await rconTry(bot, rcon, `spreadplayers ${located.x} ${located.z} 0 ${VILLAGE_LAND_RADIUS} false ${bot.username}`);
      hops.push(`attempt ${attempts}: village at ${located.x},${located.z} (${located.raw}) -> ${res}`);
      if (!res || !/^Spread/.test(res)) continue;
      await sleep(VILLAGE_SETTLE_MS);
      const r = await planVillagePlacement(bot, rng, bot.entity.position);
      searchInfo = { spots_found: r.spots_found, columns_checked: r.columns_checked, rejected: r.rejected, landing: [bot.entity.position.x, bot.entity.position.y, bot.entity.position.z] };
      placed = r.placement;
    }
    this._episodeMetadata.scene_search_attempts = attempts;
    this._episodeMetadata.scene_hops = hops;
    if (placed === null) {
      console.log(`[${bot.username}] no village placement found after ${attempts} attempt(s)`);
      return null;
    }
    console.log(
      `[${bot.username}] village placement: A ${JSON.stringify(placed.posA)} cov ${placed.coverageA.toFixed(2)} B ${JSON.stringify(placed.posB)} cov ${placed.coverageB.toFixed(2)} ` +
        `dots ${placed.dots.a.toFixed(2)}/${placed.dots.b.toFixed(2)} after ${attempts} attempt(s)`,
    );
    const sweeps = [sweepPlan(rng), sweepPlan(rng)];
    const names = [bot.username, args.other_bot_name];
    const bots = {};
    [[names[0], placed.posA, placed.yawA, sweeps[0], placed.hitA, placed.coverageA, placed.builtA, placed.skyA, placed.per_yaw_a],
      [names[1], placed.posB, placed.yawB, sweeps[1], placed.hitB, placed.coverageB, placed.builtB, placed.skyB, placed.per_yaw_b]].forEach(
      ([name, pos, yaw0, sw, hit, cov, built, sky, perYaw]) => {
        bots[name] = { name, pos: [pos.x, pos.y, pos.z], yaw0, ...sw, look_hit: hit, ray_coverage: cov, built_fraction: built, sky_light: sky, ray_hits_per_yaw: perYaw };
      });
    const dA = viewDir(placed.yawA);
    const dB = viewDir(placed.yawB);
    const startOverlap = {
      view_angle_deg: Math.acos(Math.max(-1, Math.min(1, dA.x * dB.x + dA.z * dB.z))) / DEG,
      bot_distance: Math.hypot(placed.posA.x - placed.posB.x, placed.posA.z - placed.posB.z),
      baseline_forward_dot: placed.dots,
      shared_target: placed.hitA,
      placement_tries: placed.tries,
    };

    // Entities out (villagers, golems, animals, item frames), then place
    // both bots. No block of the village is changed.
    const killCmd = `execute positioned ${Math.floor(placed.posA.x)} ${Math.floor(placed.posA.y)} ${Math.floor(placed.posA.z)} run kill @e[type=!minecraft:player,distance=..64]`;
    await rconTry(bot, rcon, killCmd);
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
    await rconTry(bot, rcon, killCmd);
    const exported = exportBlocks(bot, [placed.posA, placed.posB]);
    console.log(`[${bot.username}] village export: ${exported.blocks.length} blocks over ${exported.columns} columns (unloaded ${exported.unloaded})`);

    return {
      scene_type: "village",
      centre: { x: (placed.posA.x + placed.posB.x) / 2, y: Math.min(placed.posA.y, placed.posB.y), z: (placed.posA.z + placed.posB.z) / 2 },
      village: {
        located: located,
        search: searchInfo,
        ray_check: { step_deg: RAY_STEP_DEG, pitches_deg: RAY_PITCHES_DEG, max_dist: RAY_MAX_DIST, min_coverage: RAY_MIN_COVERAGE,
          min_built_fraction: MIN_BUILT_FRACTION, min_sky_light: MIN_SKY_LIGHT, open_above_blocks: OPEN_ABOVE_BLOCKS },
        export_radius: VILLAGE_EXPORT_RADIUS,
        export_y_range: exported.y_range,
        export_columns: exported.columns,
        export_unloaded: exported.unloaded,
        block_format: "[x, y, z, name, properties?]",
        blocks: exported.blocks,
      },
      bots,
      start_overlap: startOverlap,
      attempts,
      hops,
      world_edits_for_placement: [],
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
    for (const cmd of ["gamerule randomTickSpeed 3", "gamerule doWeatherCycle true", "gamerule doTileDrops true", "gamerule doMobLoot true", "gamerule doMobSpawning true"]) {
      await rconTry(bot, rcon, cmd);
    }
    if (this._layout && this._layout.courtyard) {
      const c = this._layout.courtyard;
      const top = (this._layout.terraform && this._layout.terraform.top_block) || "grass_block";
      try {
        await rcon.send(`fill ${c.box_min[0]} ${c.box_min[1] + 1} ${c.box_min[2]} ${c.box_max[0]} ${c.box_max[1]} ${c.box_max[2]} minecraft:air`);
        await rcon.send(`fill ${c.box_min[0]} ${c.box_min[1]} ${c.box_min[2]} ${c.box_max[0]} ${c.box_min[1]} ${c.box_max[2]} minecraft:${top}`);
      } catch (err) {
        console.log(`[${bot.username}] courtyard cleanup failed: ${err.message}`);
      }
    }
  }
}

module.exports = { SharedArenaEvalEpisode };

const seedrandom = require("seedrandom");
const { Vec3 } = require("vec3");

const { stopAll } = require("../../primitives/movement");
const { decidePrimaryBot } = require("../../utils/coordination");
const { sleep } = require("../../utils/helpers");
const { BaseEpisode } = require("../base-episode");
const { sceneUtils } = require("./traj-segments-eval-episode");

const {
  isAir,
  isPassable,
  isStandable,
  isOpenColumn,
  isFlatGround,
  blockAtXYZ,
  isViewGood,
  EYE_HEIGHT,
} = sceneUtils;

// Fixed base seed; the per-episode seed is derived from (base seed, instance
// id, episode number) so both bots agree on the plan.
const WORLD_MOD_BASE_SEED = 20261008;

// Colour-contrasting, non-gravity, pickaxe-mineable blocks.
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

// Timing (physics ticks; the recording runs at ~20 fps so 1 tick ~ 1 frame).
// Every event is preceded and followed by HOLD_TICKS ticks during which
// neither bot moves or turns. The lead (before the first event) is the first
// event's pre-hold; between events the actor may turn to aim at the next cell
// (TURN_MAX_TICKS + 1 settle tick) between the previous post-hold and the
// next pre-hold. The window (lead start .. tail end) is padded to
// WINDOW_TARGET_TICKS when the events finish early.
const HOLD_TICKS = 16;
// Extra static ticks after the last post-hold, so the observer's recording
// (whose frames are stamped a few frames off the actor's clock) still holds
// 15 frames after the last event when the recording stops.
const TAIL_TICKS = 6;
const TURN_MAX_TICKS = 6;
const TURN_MIN_OMEGA_DEG_PER_SEC = 60;
const WINDOW_TARGET_TICKS = 112;
const EVENT_TIMEOUT_MS = 3000;
// Concrete (hardness 1.8) with a plain diamond pickaxe breaks in 7 ticks
// (no Haste: normal-game digging). To keep 3-event episodes inside the
// 105-129 frame window they hold at most one break; see buildEventTypes.

// Geometry. The target area is a 3x3 patch of flat ground; event cells are
// the air cells directly above it. Distances are horizontal, to the patch
// centre.
const PATCH_HALF = 1;
const ACTOR_DIST = [3.0, 4.5];
const OBSERVER_DIST = [3.0, 6.0];
// Eye-to-aim-point distance for every event cell of the actor (server block
// interaction range is raised to INTERACTION_RANGE; vanilla default 4.5).
const ACTOR_CELL_REACH = [2.2, 5.0];
const INTERACTION_RANGE = 6.0;
// Angle between the two lines of sight (A->centre, B->centre). Each episode
// draws one band: the narrow band feeds the two-view-consistency metric
// (cross-view matching is hard at wide angles), the wide band the
// state-propagation metric. The wide band is drawn a little more often
// because part of its events cannot be matched by the evaluator and get
// excluded from scoring. The angle is recorded per episode as
// view_angle_deg / view_angle_band.
const VIEW_ANGLE_BANDS_DEG = [
  [10, 30],
  [30, 90],
];
const WIDE_BAND_PROBABILITY = 0.54;
const MIN_BOT_SEPARATION = 2.5;
const MIN_OBSERVER_CELL_DISTANCE = 1.5;
// A bot's body (0.6 wide, 1.8 tall) must stay this far from the other bot's
// lines of sight to the event cells.
const BODY_CLEARANCE = 0.6;
const BODY_HEIGHT = 1.9;
// Mobs are removed this far around the target before recording.
const MOB_CLEAR_RADIUS = 48;
// Largest angular jump between consecutive aims (fits in TURN_MAX_TICKS).
const MAX_AIM_SEPARATION_DEG = 30;

// Scene search (see pickScene).
const SCENE_SEARCH_RADIUS = 8;
const SCENE_SEARCH_BUDGET_MS = 5000;
const SCENE_MAX_ATTEMPTS = 10;
const SCENE_HOP_RANGE = 24;
const SCENE_HOP_MAX_RANGE = 320;
const SCENE_HOP_SETTLE_MS = 4000;
const SPOT_CLEAR_RADIUS = 1;
const MAX_ACTOR_CANDIDATES = 16;
const MAX_OBSERVER_CANDIDATES = 16;
const OBSERVER_CHUNK_SETTLE_MS = 2500;

const DEG = Math.PI / 180;

// ---------------------------------------------------------------------------
// Plan
// ---------------------------------------------------------------------------

/**
 * Event types and blocks for one episode: 2 or 3 events in a random order
 * with distinct block types. A break costs ~8 ticks (7-tick dig + server
 * confirmation) against ~2 for a placement, so 3-event episodes are always
 * 2 places + 1 break, and 2-event episodes are 1+1 or 2 breaks (half each):
 * place and break then come out equal overall (1.25 each per episode) and
 * the longest script (3 events, 2 turns) stays inside the frame window.
 */
function buildEventTypes(rng) {
  const numEvents = rng() < 0.5 ? 2 : 3;
  const numPlace = numEvents === 3 ? 2 : rng() < 0.5 ? 1 : 0;
  const types = [];
  for (let i = 0; i < numEvents; i++) types.push(i < numPlace ? "place" : "break");
  for (let k = types.length - 1; k > 0; k--) {
    const j = Math.floor(rng() * (k + 1));
    [types[k], types[j]] = [types[j], types[k]];
  }
  const blocks = BLOCK_TYPES.slice();
  for (let k = blocks.length - 1; k > 0; k--) {
    const j = Math.floor(rng() * (k + 1));
    [blocks[k], blocks[j]] = [blocks[j], blocks[k]];
  }
  return types.map((type, idx) => ({ idx, type, block: blocks[idx] }));
}

function shuffled(arr, rng) {
  const a = arr.slice();
  for (let k = a.length - 1; k > 0; k--) {
    const j = Math.floor(rng() * (k + 1));
    [a[k], a[j]] = [a[j], a[k]];
  }
  return a;
}

// ---------------------------------------------------------------------------
// Geometry helpers (mineflayer convention: yaw 0 = north = -z, yaw grows
// counter-clockwise, pitch positive = up)
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

function dist3(a, b) {
  return Math.hypot(b.x - a.x, b.y - a.y, b.z - a.z);
}

function cellKey(c) {
  return `${c.x},${c.y},${c.z}`;
}

// Where the actor's crosshair rests for an event: the top face of the ground
// block for a placement (the use-click lands there), the block centre for a
// break.
function aimPointOf(ev) {
  const [x, y, z] = ev.pos;
  return ev.type === "place"
    ? { x: x + 0.5, y, z: z + 0.5 }
    : { x: x + 0.5, y: y + 0.5, z: z + 0.5 };
}

function cellCentre(c) {
  return { x: c.x + 0.5, y: c.y + 0.5, z: c.z + 0.5 };
}

// Angular separation (deg) between two look directions.
function lookSeparationDeg(a, b) {
  const v = (l) => ({
    x: -Math.sin(l.yaw) * Math.cos(l.pitch),
    y: Math.sin(l.pitch),
    z: -Math.cos(l.yaw) * Math.cos(l.pitch),
  });
  const p = v(a);
  const q = v(b);
  const d = Math.max(-1, Math.min(1, p.x * q.x + p.y * q.y + p.z * q.z));
  return Math.acos(d) / DEG;
}

/**
 * True when the segment from `from` to `to` passes through a solid block
 * (full-cube or other collidable) or through one of the `virtualSolid` cells
 * (cells that will hold a block at some point of the episode). The cell that
 * contains `to` is ignored.
 */
function rayBlocked(bot, from, to, virtualSolid, ignoreCell) {
  const d = dist3(from, to);
  const n = Math.max(1, Math.ceil(d / 0.1));
  let last = null;
  for (let i = 0; i < n; i++) {
    const t = (i / n) * d;
    const px = from.x + ((to.x - from.x) * t) / d;
    const py = from.y + ((to.y - from.y) * t) / d;
    const pz = from.z + ((to.z - from.z) * t) / d;
    const key = `${Math.floor(px)},${Math.floor(py)},${Math.floor(pz)}`;
    if (key === last) continue;
    last = key;
    if (ignoreCell && key === ignoreCell) continue;
    if (virtualSolid.has(key)) return true;
    const block = blockAtXYZ(bot, px, py, pz);
    if (block === null) return true; // unloaded: assume blocked
    if (block.boundingBox !== "empty") return true;
  }
  return false;
}

// Line of sight from an eye to an event cell: the cell centre, the actor's
// aim point and the top-face corners must be reachable without crossing
// world geometry or another plan cell (a neighbouring block must not hide
// part of the cell either).
function cellVisibleFrom(bot, eye, ev, virtualSolid) {
  const key = ev.pos.join(",");
  const [x, y, z] = ev.pos;
  const targets = [
    cellCentre({ x, y, z }),
    aimPointOf(ev),
    { x: x + 0.1, y: y + 0.9, z: z + 0.1 },
    { x: x + 0.9, y: y + 0.9, z: z + 0.1 },
    { x: x + 0.1, y: y + 0.9, z: z + 0.9 },
    { x: x + 0.9, y: y + 0.9, z: z + 0.9 },
  ];
  return targets.every((t) => !rayBlocked(bot, eye, t, virtualSolid, key));
}

// True when the segment eye -> point passes within BODY_CLEARANCE of the
// column of a bot standing at `pos` (feet), below its head height.
function rayHitsBody(eye, point, pos) {
  const d = dist3(eye, point);
  const n = Math.max(1, Math.ceil(d / 0.1));
  for (let i = 0; i <= n; i++) {
    const t = i / n;
    const px = eye.x + (point.x - eye.x) * t;
    const py = eye.y + (point.y - eye.y) * t;
    const pz = eye.z + (point.z - eye.z) * t;
    if (py < pos.y || py > pos.y + BODY_HEIGHT) continue;
    if (Math.hypot(px - pos.x, pz - pos.z) < BODY_CLEARANCE) return true;
  }
  return false;
}

function cellHiddenByBody(eye, ev, otherPos) {
  const c = cellCentre({ x: ev.pos[0], y: ev.pos[1], z: ev.pos[2] });
  return rayHitsBody(eye, c, otherPos) || rayHitsBody(eye, aimPointOf(ev), otherPos);
}

// ---------------------------------------------------------------------------
// Scene selection
// ---------------------------------------------------------------------------

// 3x3 patch of level, full-cube ground with two blocks of air (or plants,
// which are cleared before recording) above.
function isPatch(bot, cx, y, cz) {
  for (let dx = -PATCH_HALF; dx <= PATCH_HALF; dx++) {
    for (let dz = -PATCH_HALF; dz <= PATCH_HALF; dz++) {
      const x = cx + dx;
      const z = cz + dz;
      if (!isFlatGround(blockAtXYZ(bot, x, y - 1, z))) return false;
      if (!isPassable(blockAtXYZ(bot, x, y, z))) return false;
      if (!isPassable(blockAtXYZ(bot, x, y + 1, z))) return false;
    }
  }
  return true;
}

function isSpotClear(bot, pos) {
  for (let dx = -SPOT_CLEAR_RADIUS; dx <= SPOT_CLEAR_RADIUS; dx++) {
    for (let dz = -SPOT_CLEAR_RADIUS; dz <= SPOT_CLEAR_RADIUS; dz++) {
      if (!isOpenColumn(bot, pos.x + dx, pos.y, pos.z + dz)) return false;
    }
  }
  return true;
}

/**
 * Pick the event cells for an actor standing at `actorPos`: every cell must
 * be within reach and consecutive aims must be at most
 * MAX_AIM_SEPARATION_DEG apart. Returns the events with pos/aim filled, or
 * null.
 */
function chooseCells(bot, actorPos, centre, eventTypes, rng) {
  const eye = eyeOf(actorPos);
  const cells = [];
  for (let dx = -PATCH_HALF; dx <= PATCH_HALF; dx++) {
    for (let dz = -PATCH_HALF; dz <= PATCH_HALF; dz++) {
      cells.push({ x: Math.floor(centre.x) + dx, y: centre.y, z: Math.floor(centre.z) + dz });
    }
  }
  const reachable = cells.filter((c) => {
    const place = dist3(eye, { x: c.x + 0.5, y: c.y, z: c.z + 0.5 });
    const brk = dist3(eye, cellCentre(c));
    const far = Math.max(place, brk);
    const near = Math.min(place, brk);
    return far <= ACTOR_CELL_REACH[1] && near >= ACTOR_CELL_REACH[0];
  });
  if (reachable.length < eventTypes.length) return null;
  for (let attempt = 0; attempt < 30; attempt++) {
    const picked = shuffled(reachable, rng).slice(0, eventTypes.length);
    const events = eventTypes.map((ev, i) => ({
      ...ev,
      pos: [picked[i].x, picked[i].y, picked[i].z],
    }));
    let ok = true;
    for (let i = 1; i < events.length; i++) {
      const a = anglesTo(eye, aimPointOf(events[i - 1]));
      const b = anglesTo(eye, aimPointOf(events[i]));
      if (lookSeparationDeg(a, b) > MAX_AIM_SEPARATION_DEG) {
        ok = false;
        break;
      }
    }
    if (ok) return events;
  }
  return null;
}

/**
 * Search the actor's surroundings for a target patch plus two standing spots:
 * the actor A (ACTOR_DIST from the patch centre, every event cell in reach)
 * and the observer B (OBSERVER_DIST, `angleBand` degrees off A's line of
 * sight). Both see every event cell and each other unobstructed, both stand
 * on level ground at the patch height with landmarks in view.
 * @returns {{centre, actorPos, observerPos, events, candidates, rejected}|null}
 */
async function pickScene(bot, eventTypes, rng, angleBand) {
  const base = bot.entity.position.floored();
  const columns = [];
  for (let dx = -SCENE_SEARCH_RADIUS; dx <= SCENE_SEARCH_RADIUS; dx++) {
    for (let dz = -SCENE_SEARCH_RADIUS; dz <= SCENE_SEARCH_RADIUS; dz++) {
      columns.push({ dx, dz, d2: dx * dx + dz * dz });
    }
  }
  columns.sort((a, b) => a.d2 - b.d2);

  const deadline = Date.now() + SCENE_SEARCH_BUDGET_MS;
  const rejected = { patch: 0, spots: 0, cells: 0, actor_los: 0, observer: 0 };
  let candidates = 0;
  const maxH = OBSERVER_DIST[1] + 1;
  for (const col of columns) {
    if (Date.now() > deadline) break;
    const cx = base.x + col.dx;
    const cz = base.z + col.dz;
    for (let dy = 1; dy >= -2; dy--) {
      const y = base.y + dy;
      if (!isPatch(bot, cx, y, cz)) {
        rejected.patch++;
        continue;
      }
      const centre = { x: cx + 0.5, y, z: cz + 0.5 };
      // Standing spots on the patch level around the patch.
      const spots = [];
      for (let ddx = -Math.ceil(maxH); ddx <= Math.ceil(maxH); ddx++) {
        for (let ddz = -Math.ceil(maxH); ddz <= Math.ceil(maxH); ddz++) {
          const h = Math.hypot(ddx, ddz);
          if (h < OBSERVER_DIST[0] - 0.01 || h > maxH) continue;
          const pos = { x: cx + ddx + 0.5, y, z: cz + ddz + 0.5 };
          if (!isStandable(bot, pos.x, pos.y, pos.z)) continue;
          if (!isSpotClear(bot, pos)) continue;
          spots.push({ pos, h, ang: Math.atan2(ddz, ddx) });
        }
      }
      const actorSpots = spots.filter((s) => s.h >= ACTOR_DIST[0] && s.h <= ACTOR_DIST[1]);
      const observerSpots = spots.filter((s) => s.h >= OBSERVER_DIST[0] && s.h <= OBSERVER_DIST[1]);
      if (actorSpots.length === 0 || observerSpots.length === 0) {
        rejected.spots++;
        continue;
      }
      for (const a of shuffled(actorSpots, rng).slice(0, MAX_ACTOR_CANDIDATES)) {
        candidates++;
        const events = chooseCells(bot, a.pos, centre, eventTypes, rng);
        if (events === null) {
          rejected.cells++;
          continue;
        }
        const virtualSolid = new Set(events.map((ev) => ev.pos.join(",")));
        const eyeA = eyeOf(a.pos);
        if (!events.every((ev) => cellVisibleFrom(bot, eyeA, ev, virtualSolid))) {
          rejected.actor_los++;
          continue;
        }
        const yawA = anglesTo(eyeA, centre).yaw;
        if (!isViewGood(bot, a.pos, yawA)) {
          rejected.actor_los++;
          continue;
        }
        for (const b of shuffled(observerSpots, rng).slice(0, MAX_OBSERVER_CANDIDATES)) {
          const dAng = Math.abs(wrapRad(b.ang - a.ang)) / DEG;
          if (dAng < angleBand[0] || dAng > angleBand[1]) continue;
          if (Math.hypot(b.pos.x - a.pos.x, b.pos.z - a.pos.z) < MIN_BOT_SEPARATION) continue;
          if (
            events.some(
              (ev) =>
                Math.hypot(b.pos.x - ev.pos[0] - 0.5, b.pos.z - ev.pos[2] - 0.5) <
                MIN_OBSERVER_CELL_DISTANCE,
            )
          ) {
            continue;
          }
          const eyeB = eyeOf(b.pos);
          if (rayBlocked(bot, eyeA, eyeB, virtualSolid, null)) continue;
          if (!events.every((ev) => cellVisibleFrom(bot, eyeB, ev, virtualSolid))) continue;
          if (events.some((ev) => cellHiddenByBody(eyeB, ev, a.pos))) continue;
          if (events.some((ev) => cellHiddenByBody(eyeA, ev, b.pos))) continue;
          const yawB = anglesTo(eyeB, centre).yaw;
          if (!isViewGood(bot, b.pos, yawB)) continue;
          return {
            centre,
            actorPos: new Vec3(a.pos.x, a.pos.y, a.pos.z),
            observerPos: new Vec3(b.pos.x, b.pos.y, b.pos.z),
            actorDist: a.h,
            observerDist: b.h,
            viewAngleDeg: dAng,
            events,
            candidates,
            rejected,
          };
        }
        rejected.observer++;
      }
      // Patches are at most one per column; keep scanning other columns.
      break;
    }
    await sleep(0);
  }
  console.log(
    `[${bot.username}] scene search: no suitable layout among ${candidates} actor candidates, rejected by ${JSON.stringify(rejected)}`,
  );
  return null;
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

// Poll the server until the block at pos is blockName (or the timeout
// passes). Returns the time of the first confirmation, or null.
async function rconWaitBlock(rcon, pos, blockName, timeoutMs) {
  const deadline = Date.now() + timeoutMs;
  for (;;) {
    if (await rconBlockIs(rcon, pos, blockName)) return Date.now();
    if (Date.now() > deadline) return null;
    await sleep(20);
  }
}

async function rconSetBlock(rcon, pos, blockName) {
  return rcon.send(`setblock ${pos[0]} ${pos[1]} ${pos[2]} minecraft:${blockName}`);
}

async function setInteractionRange(bot, rcon, value) {
  for (const attr of [
    "minecraft:player.block_interaction_range",
    "minecraft:block_interaction_range",
  ]) {
    const res = String(await rcon.send(`attribute ${bot.username} ${attr} base set ${value}`));
    console.log(`[${bot.username}] ${attr} -> ${res}`);
    if (!/unknown|no such|incorrect|expected|invalid/i.test(res)) return true;
  }
  return false;
}

// Recipes, chat and advancement notifications: unlock every recipe for the
// bot and its camera so no "new recipe" toast can pop, silence advancement
// chat and command feedback. Once per bot process.
async function quietNotificationsOnce(bot, rcon) {
  if (bot._worldModQuietDone) return;
  bot._worldModQuietDone = true;
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

// ---------------------------------------------------------------------------
// Coordination helper
// ---------------------------------------------------------------------------

/**
 * Rendezvous on `name` inside setupEpisode: register, send, and re-send once
 * from the handler so the slower bot's listener is always triggered (the
 * same double-send pattern as the phase handlers). Resolves when the peer's
 * message arrived.
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
// Motion / action primitives
// ---------------------------------------------------------------------------

function lookState(bot) {
  return { yaw: bot.entity.yaw, pitch: bot.entity.pitch };
}

async function lookExact(bot, look, target) {
  look.yaw = target.yaw;
  look.pitch = target.pitch;
  await bot.look(target.yaw, target.pitch, true);
}

/**
 * Turn both axes at a constant rate from the current look to `target`, one
 * forced look per tick, in at most TURN_MAX_TICKS ticks (the rate rises above
 * TURN_MIN_OMEGA_DEG_PER_SEC when needed). Ends on the exact target.
 * Returns ticks spent.
 */
async function turnTo(bot, look, target) {
  const dyaw = wrapRad(target.yaw - look.yaw);
  const dpitch = target.pitch - look.pitch;
  const span = Math.max(Math.abs(dyaw), Math.abs(dpitch));
  const minStep = (TURN_MIN_OMEGA_DEG_PER_SEC * DEG) / 20;
  const n = Math.min(TURN_MAX_TICKS, Math.max(1, Math.ceil(span / minStep)));
  const y0 = look.yaw;
  const p0 = look.pitch;
  for (let i = 1; i <= n; i++) {
    look.yaw = y0 + (dyaw * i) / n;
    look.pitch = p0 + (dpitch * i) / n;
    await bot.look(look.yaw, look.pitch, true);
    await bot.waitForTicks(1);
  }
  await lookExact(bot, look, target);
  await bot.waitForTicks(1);
  return n + 1;
}

function withTimeout(promise, ms, what) {
  let timer;
  const timeout = new Promise((_, reject) => {
    timer = setTimeout(() => reject(new Error(`${what} timed out after ${ms} ms`)), ms);
  });
  return Promise.race([promise, timeout]).finally(() => clearTimeout(timer));
}

// The pickaxe stays in the main hand for the whole episode and blocks are
// placed from the off-hand, so the hotbar selection never changes while
// recording (a hotbar switch pops the item's name above the hotbar for ~2 s
// in the camera client, and that option no longer exists in 1.21).
async function equipPickaxe(bot) {
  const item = bot.inventory.items().find((i) => i.name === "diamond_pickaxe");
  if (!item) throw new Error("diamond_pickaxe not in inventory");
  if (!(bot.heldItem && bot.heldItem.name === "diamond_pickaxe")) {
    await bot.equip(item, "hand");
  }
}

async function equipFor(bot, ev) {
  if (ev.type !== "place") return;
  const off = bot.inventory.slots[45];
  if (off && off.name === ev.block) return;
  const item = bot.inventory.items().find((i) => i.name === ev.block);
  if (!item) throw new Error(`${ev.block} not in inventory`);
  await bot.equip(item, "off-hand");
}

// Execute one event with the camera untouched (forceLook 'ignore'): the actor
// already aims at the event's aim point. Resolves once the client saw the
// block change.
async function doEvent(bot, ev) {
  const [x, y, z] = ev.pos;
  const pos = new Vec3(x, y, z);
  if (ev.type === "place") {
    const ref = bot.blockAt(pos.offset(0, -1, 0));
    await withTimeout(
      bot._placeBlockWithOptions(ref, new Vec3(0, 1, 0), {
        swingArm: "left",
        offhand: true,
        forceLook: "ignore",
      }),
      EVENT_TIMEOUT_MS,
      "place",
    );
  } else {
    const block = bot.blockAt(pos);
    if (!block || block.boundingBox === "empty") {
      throw new Error(`nothing to break at ${pos}: ${block && block.name}`);
    }
    await withTimeout(bot.dig(block, "ignore"), EVENT_TIMEOUT_MS, "dig");
  }
}

// ---------------------------------------------------------------------------
// Phases
// ---------------------------------------------------------------------------

function getOnWorldModPhaseFn(
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
      "worldModPhase",
      bot.entity.position.clone(),
      episodeNum,
      "worldModPhase beginning",
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
        "worldModPhase end",
      );
    };

    if (!layout) {
      console.log(`[${bot.username}] no layout (scene search failed), stopping`);
      stopNow();
      return;
    }

    if (!episodeInstance._isActor) {
      // Observer: register for the actor's event log first (the actor sends
      // it right before stopPhase), then hold perfectly still.
      const seen = [];
      const cells = new Map(layout.events.map((ev) => [ev.pos.join(","), ev.idx]));
      const onBlockUpdate = (oldBlock, newBlock) => {
        const b = newBlock || oldBlock;
        if (!b) return;
        const idx = cells.get(`${b.position.x},${b.position.y},${b.position.z}`);
        if (idx === undefined) return;
        seen.push({
          idx,
          time_ms: Date.now(),
          old_block: oldBlock ? oldBlock.name : null,
          new_block: newBlock ? newBlock.name : null,
        });
      };
      bot.on("blockUpdate", onBlockUpdate);
      coordinator.onceEvent("worldModDone", episodeNum, (data) => {
        bot.removeListener("blockUpdate", onBlockUpdate);
        episodeInstance._episodeMetadata.events = data.events;
        episodeInstance._episodeMetadata.window_start_time_ms = data.window_start_time_ms;
        episodeInstance._episodeMetadata.window_end_time_ms = data.window_end_time_ms;
        episodeInstance._episodeMetadata.observer_block_updates = seen;
        console.log(
          `[${bot.username}] received ${data.events.length} events from actor, observed ${seen.length} block updates`,
        );
        stopNow();
      });
      await lookExact(bot, lookState(bot), anglesTo(eyeOf(bot.entity.position), layout.observerAim));
      return;
    }

    // Actor.
    const events = layout.events;
    const look = lookState(bot);
    const eyeA = eyeOf(layout.actorPos);
    const eyeB = eyeOf(layout.observerPos);
    const virtualSolid = new Set(events.map((ev) => ev.pos.join(",")));
    let windowStart = null;
    let windowEnd = null;
    try {
      await lookExact(bot, look, anglesTo(eyeA, aimPointOf(events[0])));
      await bot.waitForTicks(2);
      windowStart = Date.now();
      await bot.waitForTicks(HOLD_TICKS); // lead = first event's pre-hold

      for (const ev of events) {
        if (ev.idx > 0) {
          await equipFor(bot, ev);
          ev.turn_ticks = await turnTo(bot, look, anglesTo(eyeA, aimPointOf(ev)));
          await bot.waitForTicks(HOLD_TICKS); // pre-hold
        }
        const before = ev.type === "place" ? "air" : ev.block;
        const after = ev.type === "place" ? ev.block : "air";
        ev.held_item = bot.heldItem ? bot.heldItem.name : null;
        ev.offhand_item = bot.inventory.slots[45] ? bot.inventory.slots[45].name : null;
        ev.start_time_ms = Date.now();
        try {
          await doEvent(bot, ev);
          ev.error = null;
        } catch (err) {
          ev.error = err.message;
          console.log(`[${bot.username}] event ${ev.idx} ${ev.type} failed: ${err.message}`);
        }
        ev.client_done_time_ms = Date.now();
        // The client's own dig completion is optimistic (mineflayer clears the
        // block locally when it sends the stop-digging packet), so the event
        // ends when the server confirms the new block.
        const confirmed = await rconWaitBlock(rcon, ev.pos, after, 1000);
        ev.end_time_ms = confirmed ?? Date.now();
        const b = bot.blockAt(new Vec3(ev.pos[0], ev.pos[1], ev.pos[2]));
        ev.client_block_after = b ? b.name : null;
        ev.server_block_after_ok = confirmed !== null;
        ev.world_changed = ev.client_block_after === after && ev.server_block_after_ok;
        // Line of sight from both eyes at the event, through the world as the
        // actor's client sees it now (includes blocks placed so far).
        ev.los_clear = {
          [bot.username]: cellVisibleFrom(bot, eyeA, ev, new Set()),
          [args.other_bot_name]: cellVisibleFrom(bot, eyeB, ev, new Set()),
        };
        ev.los_clear_plan = {
          [bot.username]: cellVisibleFrom(bot, eyeA, ev, virtualSolid),
          [args.other_bot_name]: cellVisibleFrom(bot, eyeB, ev, virtualSolid),
        };
        console.log(
          `[${bot.username}] event ${ev.idx} ${ev.type} ${ev.block} at ${ev.pos} ` +
            `expected ${before}->${after}, client=${ev.client_block_after}, server_ok=${ev.server_block_after_ok}, ` +
            `${ev.end_time_ms - ev.start_time_ms} ms`,
        );
        await bot.waitForTicks(HOLD_TICKS); // post-hold
      }
      const usedTicks = Math.round((Date.now() - windowStart) / 50);
      await bot.waitForTicks(Math.max(TAIL_TICKS, WINDOW_TARGET_TICKS - usedTicks));
      windowEnd = Date.now();
    } finally {
      stopAll(bot);
    }

    episodeInstance._episodeMetadata.events = events;
    episodeInstance._episodeMetadata.window_start_time_ms = windowStart;
    episodeInstance._episodeMetadata.window_end_time_ms = windowEnd;
    coordinator.sendToOtherBot(
      "worldModDone",
      {
        events,
        window_start_time_ms: windowStart,
        window_end_time_ms: windowEnd,
      },
      episodeNum,
      "worldModPhase actor done",
    );
    stopNow();
  };
}

/**
 * World-modification eval episode. Bot A (actor) places or breaks blocks at
 * known coordinates in a flat target area while bot B (observer) watches the
 * same area from a different angle; both stand still. Checks a world model's
 * interaction fidelity (A's own view reflects A's edits) and state
 * propagation (B's view reflects them too).
 *
 * episode_info (both bots) carries `events`: idx, type (place|break), block,
 * pos [x,y,z] (block coordinates), actor, the wall-clock window of the action
 * (start_time_ms/end_time_ms -> frame_start/frame_end in postprocess),
 * server/client verification of the world change and per-bot line-of-sight.
 * `window_start_time_ms`/`window_end_time_ms` bound the static-hold window
 * (105-129 frames). The observer additionally logs the block updates its own
 * client received (`observer_block_updates`).
 * @extends BaseEpisode
 */
class WorldModEvalEpisode extends BaseEpisode {
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
    const seed = `worldModEval_${WORLD_MOD_BASE_SEED}_${args.instance_id ?? 0}_${episodeNum}`;
    const isActor = decidePrimaryBot(bot, sharedBotRng, args);
    this._isActor = isActor;
    this._layout = null;
    const actorName = isActor ? bot.username : args.other_bot_name;
    const observerName = isActor ? args.other_bot_name : bot.username;
    this._episodeMetadata = {
      world_mod_seed: seed,
      actor: actorName,
      observer: observerName,
      role: isActor ? "actor" : "observer",
      hold_ticks: HOLD_TICKS,
      scene_check_passed: false,
      events: [],
    };
    this._evalMetadata = {
      bots_chosen: [bot.username, args.other_bot_name].sort(),
      actor: actorName,
    };

    await quietNotificationsOnce(bot, rcon);

    let layoutPromise = null;
    if (!isActor) {
      // Register before the rendezvous so the actor's layout can never arrive
      // unheard.
      layoutPromise = new Promise((resolve) => {
        coordinator.onceEvent("worldModLayout", episodeNum, (data) => resolve(data));
      });
    }
    await rendezvous(coordinator, "worldModReady", episodeNum);

    let layout = null;
    if (isActor) {
      layout = await this.actorSetup(bot, rcon, episodeNum, args, botPosition, seed);
      coordinator.sendToOtherBot(
        "worldModLayout",
        { layout },
        episodeNum,
        "worldModLayout send",
      );
    } else {
      layout = (await layoutPromise).layout;
      if (layout) {
        // The actor teleported us; let the chunks around the target load.
        await sleep(OBSERVER_CHUNK_SETTLE_MS);
      }
    }

    if (!layout) {
      return { botPositionNew: botPosition, otherBotPositionNew: otherBotPosition };
    }
    this._layout = layout;
    this._episodeMetadata.scene_check_passed = true;
    this._episodeMetadata.target_centre = [layout.centre.x, layout.centre.y, layout.centre.z];
    this._episodeMetadata.actor_pos = [layout.actorPos.x, layout.actorPos.y, layout.actorPos.z];
    this._episodeMetadata.observer_pos = [
      layout.observerPos.x,
      layout.observerPos.y,
      layout.observerPos.z,
    ];
    this._episodeMetadata.actor_dist = layout.actorDist;
    this._episodeMetadata.observer_dist = layout.observerDist;
    this._episodeMetadata.view_angle_deg = layout.viewAngleDeg;
    this._episodeMetadata.view_angle_band = layout.viewAngleBand;
    this._episodeMetadata.scene_search_attempts = layout.attempts;
    this._episodeMetadata.events = layout.events;
    this._evalMetadata.num_events = layout.events.length;
    this._evalMetadata.event_types = layout.events.map((e) => e.type);
    this._evalMetadata.view_angle_band = layout.viewAngleBand;

    // The caller aims the camera at otherBotPositionNew with a feet-based
    // look; shift the aim point down by the eye height so the resulting
    // yaw/pitch are the eye-based ones.
    const aim = isActor ? aimPointOf(layout.events[0]) : layout.observerAim;
    const myPos = isActor ? layout.actorPos : layout.observerPos;
    return {
      botPositionNew: new Vec3(myPos.x, myPos.y, myPos.z),
      otherBotPositionNew: new Vec3(aim.x, aim.y - EYE_HEIGHT, aim.z),
    };
  }

  async actorSetup(bot, rcon, episodeNum, args, botPosition, seed) {
    const rng = seedrandom(seed);
    const eventTypes = buildEventTypes(rng);
    const angleBand = VIEW_ANGLE_BANDS_DEG[rng() < WIDE_BAND_PROBABILITY ? 1 : 0];
    this._episodeMetadata.view_angle_band = angleBand;
    console.log(
      `[${bot.username}] Episode ${episodeNum}: actor, view angle ${angleBand[0]}-${angleBand[1]} deg, ` +
        `events=${eventTypes.map((e) => `${e.type}:${e.block}`).join(" | ")}`,
    );

    let scene = null;
    let attempts = 0;
    while (scene === null && attempts < SCENE_MAX_ATTEMPTS) {
      if (attempts > 0) {
        const range = Math.min(SCENE_HOP_RANGE * 2 ** (attempts - 1), SCENE_HOP_MAX_RANGE);
        const res = await rcon.send(
          `spreadplayers ${Math.floor(botPosition.x)} ${Math.floor(botPosition.z)} 0 ${range} false ${bot.username}`,
        );
        console.log(`[${bot.username}] scene hop ${attempts}: ${res}`);
        await sleep(SCENE_HOP_SETTLE_MS);
      }
      attempts++;
      scene = await pickScene(bot, eventTypes, rng, angleBand);
    }
    if (scene === null) {
      this._episodeMetadata.scene_search_attempts = attempts;
      return null;
    }
    const { centre, actorPos, observerPos, events } = scene;
    console.log(
      `[${bot.username}] scene: centre (${centre.x}, ${centre.y}, ${centre.z}) actor ${actorPos} (${scene.actorDist.toFixed(2)}) ` +
        `observer ${observerPos} (${scene.observerDist.toFixed(2)}) angle ${scene.viewAngleDeg.toFixed(1)} deg ` +
        `after ${scene.candidates} candidates, ${attempts} attempt(s); cells ${events.map((e) => e.pos.join(",")).join(" ")}`,
    );

    // Place both bots.
    for (const [name, pos] of [
      [bot.username, actorPos],
      [args.other_bot_name, observerPos],
    ]) {
      const res = await rcon.send(`tp ${name} ${pos.x} ${pos.y} ${pos.z}`);
      console.log(`[${bot.username}] tp ${name} -> ${String(res).trim()}`);
    }
    await sleep(1500);
    if (bot.entity.position.distanceTo(actorPos) > 0.3) {
      console.log(`[${bot.username}] actor not at scene start (${bot.entity.position}), giving up`);
      return null;
    }

    // Clear plants off the patch, remove mobs around it (no loot, no new
    // spawns), then pre-place the blocks that will be broken and make sure
    // placement cells are empty. Verify through the server.
    const cx = Math.floor(centre.x);
    const cz = Math.floor(centre.z);
    for (const cmd of [
      `fill ${cx - PATCH_HALF} ${centre.y} ${cz - PATCH_HALF} ${cx + PATCH_HALF} ${centre.y + 1} ${cz + PATCH_HALF} minecraft:air`,
      "gamerule doMobLoot false",
      "gamerule doMobSpawning false",
      `execute positioned ${cx} ${centre.y} ${cz} run kill @e[type=!minecraft:player,distance=..${MOB_CLEAR_RADIUS}]`,
    ]) {
      const res = await rcon.send(cmd);
      console.log(`[${bot.username}] ${cmd} -> ${String(res).trim()}`);
    }
    for (const ev of events) {
      ev.actor = bot.username;
      ev.aim = (({ x, y, z }) => [x, y, z])(aimPointOf(ev));
      const want = ev.type === "break" ? ev.block : "air";
      const res = await rconSetBlock(rcon, ev.pos, want);
      console.log(`[${bot.username}] setblock ${ev.pos} ${want} -> ${String(res).trim()}`);
    }
    await sleep(500);
    for (const ev of events) {
      const want = ev.type === "break" ? ev.block : "air";
      ev.server_block_before_ok = await rconBlockIs(rcon, ev.pos, want);
      if (!ev.server_block_before_ok) {
        console.log(`[${bot.username}] pre-state check failed for event ${ev.idx} (${want} at ${ev.pos})`);
      }
    }

    // Actor tools: pickaxe, the blocks to place, a longer interaction range,
    // and no item drops from the breaks.
    await rcon.send(`clear ${bot.username}`);
    await rcon.send(`give ${bot.username} minecraft:diamond_pickaxe 1`);
    for (const block of new Set(events.filter((e) => e.type === "place").map((e) => e.block))) {
      await rcon.send(`give ${bot.username} minecraft:${block} 16`);
    }
    this._episodeMetadata.interaction_range_set = await setInteractionRange(
      bot,
      rcon,
      INTERACTION_RANGE,
    );
    await rcon.send("gamerule doTileDrops false");
    await sleep(800);
    await equipPickaxe(bot);
    await equipFor(bot, events.find((e) => e.type === "place") ?? events[0]);

    return {
      centre,
      actorPos,
      observerPos,
      observerAim: cellCentre({ x: Math.floor(centre.x), y: centre.y, z: Math.floor(centre.z) }),
      actorDist: scene.actorDist,
      observerDist: scene.observerDist,
      viewAngleDeg: scene.viewAngleDeg,
      viewAngleBand: angleBand,
      attempts,
      events,
    };
  }

  async entryPoint(bot, rcon, sharedBotRng, coordinator, iterationID, episodeNum, args) {
    coordinator.onceEvent(
      "worldModPhase",
      episodeNum,
      getOnWorldModPhaseFn(bot, rcon, sharedBotRng, coordinator, episodeNum, this, args),
    );
    coordinator.sendToOtherBot(
      "worldModPhase",
      bot.entity.position.clone(),
      episodeNum,
      "entryPoint end",
    );
  }

  async tearDownEpisode(bot, rcon, sharedBotRng, coordinator, episodeNum, args) {
    if (!this._isActor) return;
    for (const cmd of [
      "gamerule doTileDrops true",
      "gamerule doMobLoot true",
      "gamerule doMobSpawning true",
    ]) {
      try {
        await rcon.send(cmd);
      } catch (err) {
        console.log(`[${bot.username}] teardown ${cmd} failed: ${err.message}`);
      }
    }
    await setInteractionRange(bot, rcon, 4.5);
    if (this._layout) {
      for (const ev of this._layout.events) {
        try {
          await rconSetBlock(rcon, ev.pos, "air");
        } catch (err) {
          console.log(`[${bot.username}] cleanup setblock failed: ${err.message}`);
        }
      }
    }
  }
}

module.exports = { WorldModEvalEpisode };

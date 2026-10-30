const seedrandom = require("seedrandom");
const { Vec3 } = require("vec3");

const { stopAll } = require("../../primitives/movement");
const { decidePrimaryBot } = require("../../utils/coordination");
const { sleep } = require("../../utils/helpers");
const { BaseEpisode } = require("../base-episode");
const { worldModUtils } = require("./world-mod-eval-episode");

const {
  pickScene,
  aimPointOf,
  cellCentre,
  cellVisibleFrom,
  eyeOf,
  anglesTo,
  rconBlockIs,
  rconWaitBlock,
  rconSetBlock,
  setInteractionRange,
  quietNotificationsOnce,
  rendezvous,
  lookState,
  lookExact,
  withTimeout,
  BLOCK_TYPES,
  VIEW_ANGLE_BANDS_DEG,
  WIDE_BAND_PROBABILITY,
  INTERACTION_RANGE,
  MOB_CLEAR_RADIUS,
  PATCH_HALF,
  SCENE_MAX_ATTEMPTS,
  SCENE_HOP_RANGE,
  SCENE_HOP_MAX_RANGE,
  SCENE_HOP_SETTLE_MS,
  OBSERVER_CHUNK_SETTLE_MS,
  EYE_HEIGHT,
} = worldModUtils;

// Fixed base seed; the per-episode seed is derived from (base seed, instance
// id, episode number) so both bots agree on the plan.
const WORLD_MOD_SINGLE_BASE_SEED = 20261030;

// Timing (physics ticks; the recording runs at ~20 fps so 1 tick ~ 1 frame).
// Both bots stand still for PRE_HOLD_TICKS, the actor performs the single
// interaction (a placement takes ~2 ticks, a break ~8-10: 7-tick dig plus
// server confirmation), then both keep still for at least POST_MIN_TICKS,
// the window being padded to WINDOW_TARGET_TICKS when that is longer. The
// window (lead start .. end) thus stays inside the 105-129 frame convention
// while leaving >= 80 frames after the event.
const PRE_HOLD_TICKS = 16;
const POST_MIN_TICKS = 86;
const WINDOW_TARGET_TICKS = 112;
const EVENT_TIMEOUT_MS = 3000;
const CONCRETE_GIVE_COUNT = 16;

// ---------------------------------------------------------------------------
// Event
// ---------------------------------------------------------------------------

// Place from the MAIN hand (the item is visible in the actor's view and the
// hotbar never changes), or dig with the pickaxe held in the main hand.
async function doEvent(bot, ev) {
  const [x, y, z] = ev.pos;
  const pos = new Vec3(x, y, z);
  if (ev.type === "place") {
    const ref = bot.blockAt(pos.offset(0, -1, 0));
    await withTimeout(
      bot._placeBlockWithOptions(ref, new Vec3(0, 1, 0), {
        swingArm: "right",
        offhand: false,
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

// Equip `itemName` in the main hand and make sure the off-hand is empty.
async function equipMainHandOnly(bot, itemName) {
  const item = bot.inventory.items().find((i) => i.name === itemName);
  if (!item) throw new Error(`${itemName} not in inventory`);
  if (!(bot.heldItem && bot.heldItem.name === itemName)) {
    await bot.equip(item, "hand");
  }
  if (bot.inventory.slots[45]) {
    await bot.unequip("off-hand");
  }
  return {
    held_item: bot.heldItem ? bot.heldItem.name : null,
    offhand_item: bot.inventory.slots[45] ? bot.inventory.slots[45].name : null,
    quick_bar_slot: bot.quickBarSlot,
  };
}

// ---------------------------------------------------------------------------
// Phase
// ---------------------------------------------------------------------------

function getOnPhaseFn(bot, rcon, sharedBotRng, coordinator, episodeNum, episodeInstance, args) {
  return async () => {
    coordinator.sendToOtherBot("worldModSinglePhase", bot.entity.position.clone(), episodeNum, "worldModSinglePhase beginning");
    const layout = episodeInstance._layout;
    const stopNow = () => {
      coordinator.onceEvent(
        "stopPhase",
        episodeNum,
        episodeInstance.getOnStopPhaseFn(bot, rcon, sharedBotRng, coordinator, args.other_bot_name, episodeNum, args),
      );
      coordinator.sendToOtherBot("stopPhase", bot.entity.position.clone(), episodeNum, "worldModSinglePhase end");
    };

    if (!layout) {
      console.log(`[${bot.username}] no layout (scene search failed), stopping`);
      stopNow();
      return;
    }
    const ev = layout.event;

    if (!episodeInstance._isActor) {
      // Observer: log the block updates of the target cell, hold still,
      // wait for the actor's report.
      const seen = [];
      const key = ev.pos.join(",");
      const onBlockUpdate = (oldBlock, newBlock) => {
        const b = newBlock || oldBlock;
        if (!b || `${b.position.x},${b.position.y},${b.position.z}` !== key) return;
        seen.push({ idx: 0, time_ms: Date.now(), old_block: oldBlock ? oldBlock.name : null, new_block: newBlock ? newBlock.name : null });
      };
      bot.on("blockUpdate", onBlockUpdate);
      coordinator.onceEvent("worldModSingleDone", episodeNum, (data) => {
        bot.removeListener("blockUpdate", onBlockUpdate);
        const meta = episodeInstance._episodeMetadata;
        meta.events = [data.event];
        meta.event = data.event;
        meta.window_start_time_ms = data.window_start_time_ms;
        meta.window_end_time_ms = data.window_end_time_ms;
        meta.observer_block_updates = seen;
        console.log(`[${bot.username}] received the event from the actor, observed ${seen.length} block updates`);
        stopNow();
      });
      await lookExact(bot, lookState(bot), anglesTo(eyeOf(bot.entity.position), layout.observerAim));
      return;
    }

    // Actor: aim once, hold, act, hold.
    const look = lookState(bot);
    const eyeA = eyeOf(layout.actorPos);
    const eyeB = eyeOf(layout.observerPos);
    let windowStart = null;
    let windowEnd = null;
    try {
      await lookExact(bot, look, anglesTo(eyeA, aimPointOf(ev)));
      await bot.waitForTicks(2);
      windowStart = Date.now();
      await bot.waitForTicks(PRE_HOLD_TICKS);

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
        console.log(`[${bot.username}] ${ev.type} failed: ${err.message}`);
      }
      ev.client_done_time_ms = Date.now();
      const confirmed = await rconWaitBlock(rcon, ev.pos, after, 1000);
      ev.end_time_ms = confirmed ?? Date.now();
      const b = bot.blockAt(new Vec3(ev.pos[0], ev.pos[1], ev.pos[2]));
      ev.client_block_after = b ? b.name : null;
      ev.server_block_after_ok = confirmed !== null;
      ev.world_changed = ev.client_block_after === after && ev.server_block_after_ok;
      ev.held_item_after = bot.heldItem ? bot.heldItem.name : null;
      ev.offhand_item_after = bot.inventory.slots[45] ? bot.inventory.slots[45].name : null;
      ev.los_clear = {
        [bot.username]: cellVisibleFrom(bot, eyeA, ev, new Set()),
        [args.other_bot_name]: cellVisibleFrom(bot, eyeB, ev, new Set()),
      };
      console.log(
        `[${bot.username}] ${ev.type} ${ev.block} at ${ev.pos} expected ${before}->${after}, client=${ev.client_block_after}, ` +
          `server_ok=${ev.server_block_after_ok}, ${ev.end_time_ms - ev.start_time_ms} ms, held ${ev.held_item} / off-hand ${ev.offhand_item}`,
      );
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
      "worldModSingleDone",
      { event: ev, window_start_time_ms: windowStart, window_end_time_ms: windowEnd },
      episodeNum,
      "worldModSinglePhase actor done",
    );
    stopNow();
  };
}

/**
 * Single-interaction world-modification eval episode. Bot A (actor) places
 * one concrete block or breaks one pre-placed concrete block in a flat
 * target area while bot B (observer) watches from another angle; both
 * stand still before and after the event, the actor holds the item in its
 * MAIN hand for the whole episode (concrete for a placement, the pickaxe
 * for a break), the off-hand is empty and the hotbar never changes.
 *
 * episode_info carries `event` (and `events` = [event] for the shared
 * postprocess): type (place|break), block, pos, aim, timing
 * (start_time_ms/end_time_ms -> frame_start/frame_end), server/client
 * verification, held / off-hand items, line of sight; the window
 * (window_start_time_ms/window_end_time_ms -> window_start_frame/
 * window_end_frame, the suggested window), poses, view angle and band.
 * @extends BaseEpisode
 */
class WorldModSingleEvalEpisode extends BaseEpisode {
  static WORKS_IN_NON_FLAT_WORLD = true;
  static INIT_MIN_BOTS_DISTANCE = 4;
  static INIT_MAX_BOTS_DISTANCE = 8;

  async setupEpisode(bot, rcon, sharedBotRng, coordinator, episodeNum, args, botPosition, otherBotPosition) {
    const seed = `worldModSingleEval_${WORLD_MOD_SINGLE_BASE_SEED}_${args.instance_id ?? 0}_${episodeNum}`;
    const isActor = decidePrimaryBot(bot, sharedBotRng, args);
    this._isActor = isActor;
    this._layout = null;
    const actorName = isActor ? bot.username : args.other_bot_name;
    const observerName = isActor ? args.other_bot_name : bot.username;
    this._episodeMetadata = {
      world_mod_single_seed: seed,
      actor: actorName,
      observer: observerName,
      role: isActor ? "actor" : "observer",
      pre_hold_ticks: PRE_HOLD_TICKS,
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
        coordinator.onceEvent("worldModSingleLayout", episodeNum, (data) => resolve(data));
      });
    }
    await rendezvous(coordinator, "worldModSingleReady", episodeNum);

    let layout = null;
    if (isActor) {
      layout = await this.actorSetup(bot, rcon, episodeNum, args, botPosition, seed);
      coordinator.sendToOtherBot("worldModSingleLayout", { layout }, episodeNum, "worldModSingleLayout send");
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
      interaction: layout.event.type,
      block: layout.event.block,
      target_cell: layout.event.pos,
      held_item: layout.held_item,
      target_centre: [layout.centre.x, layout.centre.y, layout.centre.z],
      actor_pos: [layout.actorPos.x, layout.actorPos.y, layout.actorPos.z],
      observer_pos: [layout.observerPos.x, layout.observerPos.y, layout.observerPos.z],
      actor_dist: layout.actorDist,
      observer_dist: layout.observerDist,
      view_angle_deg: layout.viewAngleDeg,
      view_angle_band: layout.viewAngleBand,
      scene_search_attempts: layout.attempts,
      events: [layout.event],
      event: layout.event,
      suggested_window: "window_start_frame .. window_end_frame (16 static frames, the event, >= 80 static frames)",
    });
    Object.assign(this._evalMetadata, { interaction: layout.event.type, block: layout.event.block, view_angle_band: layout.viewAngleBand });

    // The caller aims the camera at otherBotPositionNew with a feet-based
    // look; shift the aim point down by the eye height so the resulting
    // yaw/pitch are the eye-based ones.
    const aim = isActor ? aimPointOf(layout.event) : layout.observerAim;
    const myPos = isActor ? layout.actorPos : layout.observerPos;
    return {
      botPositionNew: new Vec3(myPos.x, myPos.y, myPos.z),
      otherBotPositionNew: new Vec3(aim.x, aim.y - EYE_HEIGHT, aim.z),
    };
  }

  async actorSetup(bot, rcon, episodeNum, args, botPosition, seed) {
    const rng = seedrandom(seed);
    // Place / break alternate with (instance + episode) parity: a 2-world x
    // 5-episode trial gets five of each.
    const type = ((args.instance_id ?? 0) + episodeNum) % 2 === 0 ? "place" : "break";
    const block = BLOCK_TYPES[Math.floor(rng() * BLOCK_TYPES.length)];
    const eventTypes = [{ idx: 0, type, block }];
    const angleBand = VIEW_ANGLE_BANDS_DEG[rng() < WIDE_BAND_PROBABILITY ? 1 : 0];
    this._episodeMetadata.view_angle_band = angleBand;
    console.log(`[${bot.username}] Episode ${episodeNum}: actor, ${type} ${block}, view angle ${angleBand[0]}-${angleBand[1]} deg`);

    let scene = null;
    let attempts = 0;
    while (scene === null && attempts < SCENE_MAX_ATTEMPTS) {
      if (attempts > 0) {
        const range = Math.min(SCENE_HOP_RANGE * 2 ** (attempts - 1), SCENE_HOP_MAX_RANGE);
        const res = await rcon.send(`spreadplayers ${Math.floor(botPosition.x)} ${Math.floor(botPosition.z)} 0 ${range} false ${bot.username}`);
        console.log(`[${bot.username}] scene hop ${attempts}: ${res}`);
        await sleep(SCENE_HOP_SETTLE_MS);
      }
      attempts++;
      scene = await pickScene(bot, eventTypes, rng, angleBand);
    }
    this._episodeMetadata.scene_search_attempts = attempts;
    if (scene === null) return null;
    const { centre, actorPos, observerPos, events } = scene;
    const ev = events[0];
    console.log(
      `[${bot.username}] scene: centre (${centre.x}, ${centre.y}, ${centre.z}) actor ${actorPos} (${scene.actorDist.toFixed(2)}) ` +
        `observer ${observerPos} (${scene.observerDist.toFixed(2)}) angle ${scene.viewAngleDeg.toFixed(1)} deg after ${attempts} attempt(s); cell ${ev.pos.join(",")}`,
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

    // Clear plants off the patch, remove mobs, pre-place the block to break /
    // make sure the placement cell is empty, verify through the server.
    const cx = Math.floor(centre.x);
    const cz = Math.floor(centre.z);
    for (const cmd of [
      `fill ${cx - PATCH_HALF} ${centre.y} ${cz - PATCH_HALF} ${cx + PATCH_HALF} ${centre.y + 1} ${cz + PATCH_HALF} minecraft:air`,
      "gamerule doMobLoot false",
      "gamerule doMobSpawning false",
      "gamerule doTileDrops false",
      `execute positioned ${cx} ${centre.y} ${cz} run kill @e[type=!minecraft:player,distance=..${MOB_CLEAR_RADIUS}]`,
    ]) {
      const res = await rcon.send(cmd);
      console.log(`[${bot.username}] ${cmd} -> ${String(res).trim()}`);
    }
    ev.actor = bot.username;
    ev.aim = (({ x, y, z }) => [x, y, z])(aimPointOf(ev));
    const want = ev.type === "break" ? ev.block : "air";
    await rconSetBlock(rcon, ev.pos, want);
    await sleep(500);
    ev.server_block_before_ok = await rconBlockIs(rcon, ev.pos, want);
    if (!ev.server_block_before_ok) console.log(`[${bot.username}] pre-state check failed (${want} at ${ev.pos})`);

    // One item only, in the main hand, for the whole episode.
    const itemName = ev.type === "place" ? ev.block : "diamond_pickaxe";
    await rcon.send(`clear ${bot.username}`);
    await rcon.send(`give ${bot.username} minecraft:${itemName} ${ev.type === "place" ? CONCRETE_GIVE_COUNT : 1}`);
    this._episodeMetadata.interaction_range_set = await setInteractionRange(bot, rcon, INTERACTION_RANGE);
    await sleep(800);
    const hands = await equipMainHandOnly(bot, itemName);
    console.log(`[${bot.username}] hands: ${JSON.stringify(hands)}`);
    if (hands.held_item !== itemName || hands.offhand_item !== null) {
      console.log(`[${bot.username}] could not arm the main hand only, giving up`);
      return null;
    }

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
      event: ev,
      held_item: hands.held_item,
      hands,
    };
  }

  async entryPoint(bot, rcon, sharedBotRng, coordinator, iterationID, episodeNum, args) {
    coordinator.onceEvent("worldModSinglePhase", episodeNum, getOnPhaseFn(bot, rcon, sharedBotRng, coordinator, episodeNum, this, args));
    coordinator.sendToOtherBot("worldModSinglePhase", bot.entity.position.clone(), episodeNum, "entryPoint end");
  }

  async tearDownEpisode(bot, rcon, sharedBotRng, coordinator, episodeNum, args) {
    if (!this._isActor) return;
    for (const cmd of ["gamerule doTileDrops true", "gamerule doMobLoot true", "gamerule doMobSpawning true"]) {
      try {
        await rcon.send(cmd);
      } catch (err) {
        console.log(`[${bot.username}] teardown ${cmd} failed: ${err.message}`);
      }
    }
    await setInteractionRange(bot, rcon, 4.5);
    if (this._layout) {
      try {
        await rconSetBlock(rcon, this._layout.event.pos, "air");
      } catch (err) {
        console.log(`[${bot.username}] cleanup setblock failed: ${err.message}`);
      }
    }
  }
}

module.exports = { WorldModSingleEvalEpisode };

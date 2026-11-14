const seedrandom = require("seedrandom");
const { Vec3 } = require("vec3");

const { stopAll } = require("../../primitives/movement");
const { decidePrimaryBot } = require("../../utils/coordination");
const { sleep } = require("../../utils/helpers");
const { BaseEpisode } = require("../base-episode");
const { sceneUtils } = require("./traj-segments-eval-episode");
const { worldModUtils } = require("./world-mod-eval-episode");
const { modScriptUtils } = require("./world-mod-revisit-eval-episode");

const { isStandable, isViewGood } = sceneUtils;
const {
  rconSetBlock,
  quietNotificationsOnce,
  rendezvous,
  shuffled,
  BLOCK_TYPES,
  SCENE_MAX_ATTEMPTS,
  SCENE_HOP_RANGE,
  SCENE_HOP_MAX_RANGE,
  SCENE_HOP_SETTLE_MS,
  OBSERVER_CHUNK_SETTLE_MS,
  DEG,
} = worldModUtils;
const {
  rconLog,
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
} = modScriptUtils;

// ---------------------------------------------------------------------------
// worldModOffscreenEval ("DSP", off-screen state propagation, two agents).
// Target, interaction and frame-stepped script: see the header of
// world-mod-revisit-eval-episode.js.
//
// Layout (T = the target column, angles in the horizontal plane):
//   A (actor)    2.8-3.7 blocks from T, faces T (pitch 0).
//   B (observer) 3.5-6 blocks from T. B starts with its heading 90 deg off
//                the direction to T, so T is 90 deg off its view centre
//                (half the horizontal FOV is 51.2 deg), and turns 90 deg
//                (3 deg per frame) onto T.
//   A stands on the side of the line B->T that B turns towards: B's start
//   heading is (90 + alpha) deg away from A, so B sees neither T nor A (nor
//   A's arm swing / break particles) until it turns; at the end A stands
//   alpha = 18-42 deg beside T in B's view. B is at least 64 deg off A's
//   view centre: A's recording never shows B.
//
// Script (recorded frames, 20 fps):
//   static_pre  10   nobody moves
//   interact    4 (place) / 10 (break)   A interacts, B does not move
//   wait        H    H = 0 / 40 / 160, nobody moves
//   b_turn      30   B turns 90 deg onto T
//   static_post 30   nobody moves
// Variants by k = (6 * instance + episode) % 12: H = HOLD_FRAMES[k % 3],
// place / break by floor(k / 3) % 2, B turns left / right by floor(k / 6).
// ---------------------------------------------------------------------------
const OFFSCREEN_BASE_SEED = 20261114;
const B_TURN_DEG = 90;
const B_TURN_FRAMES = B_TURN_DEG / DEG_PER_FRAME; // 30
const STATIC_POST_FRAMES = 30;
const B_DIST = [3.5, 6.0];
const MIN_BOT_SEPARATION = 2.5;
const A_IN_B_END_VIEW_DEG = [18, 42]; // alpha
const MIN_HIDDEN_OFFSET_DEG = 64; // B off A's view centre; A off B's start view centre
const SCENE_SEARCH_RADIUS = 8;
const SCENE_SEARCH_BUDGET_MS = 5000;
const START_DELAY_MS = 1000;

const B_OFFSETS = [];
for (let dx = -6; dx <= 6; dx++) {
  for (let dz = -6; dz <= 6; dz++) {
    const h = Math.hypot(dx, dz);
    if (h >= B_DIST[0] && h <= B_DIST[1]) B_OFFSETS.push({ dx, dz, h });
  }
}

function wrapDeg(a) {
  return ((((a + 180) % 360) + 360) % 360) - 180;
}

function offscreenSpec(instanceId, episodeNum) {
  const k = (((6 * instanceId + episodeNum) % 12) + 12) % 12;
  return {
    variant_index: k,
    hold_frames: HOLD_FRAMES[k % 3],
    interaction: Math.floor(k / 3) % 2 === 0 ? "place" : "break",
    direction: Math.floor(k / 6) % 2 === 0 ? "left" : "right", // B's turn; left = counter-clockwise = yaw grows
  };
}

function offscreenPhasePlan(spec) {
  return planPhases([
    ["static_pre", STATIC_PRE_FRAMES],
    ["interact", INTERACT_FRAMES[spec.interaction], { acting_agent: "A" }],
    ["wait", spec.hold_frames],
    ["b_turn", B_TURN_FRAMES, { turning_agent: "B" }],
    ["static_post", STATIC_POST_FRAMES],
  ]);
}

function bTurnDeg(spec, s) {
  const t = STATIC_PRE_FRAMES + INTERACT_FRAMES[spec.interaction] + spec.hold_frames;
  return s < t ? 0 : Math.min(DEG_PER_FRAME * (s - t + 1), B_TURN_DEG);
}

const bearingDeg = (from, to) => Math.atan2(-(to.x - from.x), -(to.z - from.z)) / DEG; // mineflayer yaw towards `to`

/**
 * Search around the actor for a target column and the two standing spots of
 * the layout described in the header.
 */
async function pickOffscreenScene(bot, rng, sign) {
  const base = bot.entity.position.floored();
  const columns = [];
  for (let dx = -SCENE_SEARCH_RADIUS; dx <= SCENE_SEARCH_RADIUS; dx++) {
    for (let dz = -SCENE_SEARCH_RADIUS; dz <= SCENE_SEARCH_RADIUS; dz++) columns.push({ dx, dz, d2: dx * dx + dz * dz });
  }
  columns.sort((a, b) => a.d2 - b.d2);
  const deadline = Date.now() + SCENE_SEARCH_BUDGET_MS;
  const rejected = { column: 0, actor: 0, observer: 0 };
  let candidates = 0;
  const spotOk = (p) => isStandable(bot, p.x, p.y, p.z) && isSpotClear(bot, p, 1);
  for (const col of columns) {
    if (Date.now() > deadline) break;
    const tx = base.x + col.dx;
    const tz = base.z + col.dz;
    for (let dy = 1; dy >= -2; dy--) {
      const y = base.y + dy;
      if (!columnOk(bot, tx, y, tz)) {
        rejected.column++;
        continue;
      }
      const ev = makeEvent(null, null, tx, y, tz);
      const T = { x: tx + 0.5, z: tz + 0.5 };
      for (const a of shuffled(ACTOR_OFFSETS, rng)) {
        const posA = new Vec3(tx - a.dx + 0.5, y, tz - a.dz + 0.5);
        const yawA = yawTo(posA, ev);
        if (!spotOk(posA) || !targetVisibleFrom(bot, posA, ev) || !isViewGood(bot, posA, yawA)) {
          rejected.actor++;
          continue;
        }
        for (const b of shuffled(B_OFFSETS, rng)) {
          candidates++;
          const posB = new Vec3(tx - b.dx + 0.5, y, tz - b.dz + 0.5);
          const sep = Math.hypot(posB.x - posA.x, posB.z - posA.z);
          if (sep < MIN_BOT_SEPARATION) continue;
          const yawT = bearingDeg(posB, T);
          const alphaSigned = wrapDeg(bearingDeg(posB, posA) - yawT); // > 0: A left of the line B->T
          const alpha = Math.abs(alphaSigned);
          if (alphaSigned * sign <= 0 || alpha < A_IN_B_END_VIEW_DEG[0] || alpha > A_IN_B_END_VIEW_DEG[1]) continue;
          const beta = Math.abs(wrapDeg(bearingDeg(posA, posB) - yawA / DEG));
          if (beta < MIN_HIDDEN_OFFSET_DEG) continue;
          const yawB0 = (yawT - sign * B_TURN_DEG) * DEG;
          if (!spotOk(posB) || !targetVisibleFrom(bot, posB, ev)) continue;
          if (!isViewGood(bot, posB, yawB0) || !isViewGood(bot, posB, yawT * DEG)) continue;
          return {
            ev,
            posA,
            posB,
            yawA,
            yawB0,
            yawBEnd: yawT * DEG,
            distA: a.h,
            distB: b.h,
            botDistance: sep,
            alphaDeg: alpha,
            betaDeg: beta,
            candidates,
          };
        }
        rejected.observer++;
      }
      break;
    }
    await sleep(0);
  }
  console.log(`[${bot.username}] scene search: nothing among ${candidates} candidates, rejected by ${JSON.stringify(rejected)}`);
  return null;
}

function getOnOffscreenPhaseFn(bot, rcon, sharedBotRng, coordinator, episodeNum, episodeInstance, args) {
  return async () => {
    coordinator.sendToOtherBot("worldModOffscreenPhase", bot.entity.position.clone(), episodeNum, "worldModOffscreenPhase beginning");
    const layout = episodeInstance._layout;
    const stopNow = () => {
      coordinator.onceEvent(
        "stopPhase",
        episodeNum,
        episodeInstance.getOnStopPhaseFn(bot, rcon, sharedBotRng, coordinator, args.other_bot_name, episodeNum, args),
      );
      coordinator.sendToOtherBot("stopPhase", bot.entity.position.clone(), episodeNum, "worldModOffscreenPhase end");
    };
    if (!layout) {
      console.log(`[${bot.username}] no layout (scene setup failed), stopping`);
      stopNow();
      return;
    }
    const spec = episodeInstance._spec;
    const meta = episodeInstance._episodeMetadata;
    const isActor = episodeInstance._isActor;
    const ev = layout.event;
    const phases = offscreenPhasePlan(spec);
    const windowSteps = phases[phases.length - 1].end_step + 1;
    const yaw0 = isActor ? layout.yawA : layout.yawB0;
    const seen = [];
    const key = ev.pos.join(",");
    const onBlockUpdate = (oldBlock, newBlock) => {
      const b = newBlock || oldBlock;
      if (!b || `${b.position.x},${b.position.y},${b.position.z}` !== key) return;
      seen.push({ time_ms: Date.now(), old_block: oldBlock ? oldBlock.name : null, new_block: newBlock ? newBlock.name : null });
    };
    try {
      stopAll(bot);
      await syncCamera(bot, yaw0);
      const start = bot.entity.position.clone();

      let t0Promise = null;
      let eventPromise = null;
      if (!isActor) {
        t0Promise = new Promise((resolve) => coordinator.onceEvent("worldModOffscreenT0", episodeNum, (data) => resolve(data.t0)));
        eventPromise = new Promise((resolve) => coordinator.onceEvent("worldModOffscreenDone", episodeNum, (data) => resolve(data.event)));
      }
      await rendezvous(coordinator, "worldModOffscreenGo", episodeNum);
      let t0;
      if (isActor) {
        t0 = Date.now() + START_DELAY_MS;
        coordinator.sendToOtherBot("worldModOffscreenT0", { t0 }, episodeNum, "worldModOffscreenT0 send");
      } else {
        t0 = await t0Promise;
      }
      bot.on("blockUpdate", onBlockUpdate);

      let interaction = null;
      const res = await runFrameScript(bot, {
        t0,
        windowSteps,
        yaw0,
        sign: spec.direction === "left" ? 1 : -1,
        degAt: (s) => (isActor ? 0 : bTurnDeg(spec, s)),
        triggerStep: isActor ? STATIC_PRE_FRAMES : null,
        onTrigger: () => {
          interaction = interact(bot, rcon, ev);
        },
      });
      storeScript(meta, t0, res);
      meta.drift_blocks = bot.entity.position.distanceTo(start);
      const clientEnd = new Vec3(ev.pos[0], ev.pos[1], ev.pos[2]);
      if (isActor) {
        if (interaction) await interaction;
        else ev.error = "interaction never triggered";
        ev.client_block_end = await finalState(bot, rcon, ev);
        meta.event = ev;
        coordinator.sendToOtherBot("worldModOffscreenDone", { event: ev }, episodeNum, "worldModOffscreenDone send");
      } else {
        const timeout = new Promise((resolve) => setTimeout(() => resolve(null), 10000));
        const got = await Promise.race([eventPromise, timeout]);
        meta.event = got || { ...ev, error: "actor never reported the event" };
        const b = bot.blockAt(clientEnd);
        meta.observer_client_block_end = b ? b.name : null;
      }
      meta.events = [meta.event];
      meta.own_client_block_updates = seen;
      console.log(
        `[${bot.username}] offscreen done: agent ${isActor ? "A" : "B"}, ${spec.interaction} H ${spec.hold_frames} B turns ${spec.direction}, ` +
          `${res.recorded}/${windowSteps} window frames, changed=${meta.event.world_changed}, block updates seen ${seen.length}, drift ${meta.drift_blocks.toFixed(4)}`,
      );
    } finally {
      bot.removeListener("blockUpdate", onBlockUpdate);
      stopAll(bot);
    }
    stopNow();
  };
}

/**
 * worldModOffscreenEval ("DSP"): A interacts with the target cell while B
 * looks 90 deg away; after H frames B turns onto the target. See the header
 * of this file for the layout, the script and the variants.
 *
 * episode_info at record time: eval_kind "DSP", agent (A / B), hold_frames,
 * interaction, direction (B's turn), target_cell, layout, phase_plan,
 * frame_trace, event. postprocess/world_mod_offscreen_quality.py turns them
 * into `phases`, event.effect_frame and `target_visibility` (GT frustum).
 * @extends BaseEpisode
 */
class WorldModOffscreenEvalEpisode extends BaseEpisode {
  static WORKS_IN_NON_FLAT_WORLD = true;
  static INIT_MIN_BOTS_DISTANCE = 4;
  static INIT_MAX_BOTS_DISTANCE = 8;

  async setupEpisode(bot, rcon, sharedBotRng, coordinator, episodeNum, args, botPosition, otherBotPosition) {
    const instance = args.instance_id ?? 0;
    const seed = `worldModOffscreenEval_${OFFSCREEN_BASE_SEED}_${instance}_${episodeNum}`;
    const spec = offscreenSpec(instance, episodeNum);
    const isActor = decidePrimaryBot(bot, sharedBotRng, args);
    this._spec = spec;
    this._isActor = isActor;
    this._layout = null;
    const actorName = isActor ? bot.username : args.other_bot_name;
    const observerName = isActor ? args.other_bot_name : bot.username;
    this._episodeMetadata = {
      eval_kind: "DSP",
      world_mod_offscreen_seed: seed,
      ...spec,
      agent: isActor ? "A" : "B",
      role: isActor ? "actor" : "observer",
      actor: actorName,
      observer: observerName,
      agents: { A: actorName, B: observerName },
      b_turn_deg: B_TURN_DEG,
      deg_per_frame: DEG_PER_FRAME,
      horizontal_fov_deg: HFOV_DEG,
      vertical_fov_deg: 70,
      fps: 20,
      tail_frames: TAIL_FRAMES,
      scene_check_passed: false,
      phase_plan: offscreenPhasePlan(spec),
      suggested_window: "window_start_frame .. window_end_frame (static_pre .. static_post); frames before it are the static lead, after it a static tail",
    };
    this._evalMetadata = {
      bots_chosen: [bot.username, args.other_bot_name].sort(),
      actor: actorName,
      interaction: spec.interaction,
      direction: spec.direction,
      hold_frames: spec.hold_frames,
    };

    await quietNotificationsOnce(bot, rcon);
    let layoutPromise = null;
    if (!isActor) {
      layoutPromise = new Promise((resolve) => coordinator.onceEvent("worldModOffscreenLayout", episodeNum, (data) => resolve(data)));
    }
    await rendezvous(coordinator, "worldModOffscreenReady", episodeNum);

    let layout = null;
    if (isActor) {
      try {
        layout = await this.actorSetup(bot, rcon, episodeNum, args, botPosition, seed, spec);
      } catch (err) {
        console.log(`[${bot.username}] actor setup failed: ${err.message}`);
      }
      coordinator.sendToOtherBot("worldModOffscreenLayout", { layout }, episodeNum, "worldModOffscreenLayout send");
    } else {
      layout = (await layoutPromise).layout;
      if (layout) await sleep(OBSERVER_CHUNK_SETTLE_MS);
    }
    if (!layout) return { botPositionNew: botPosition, otherBotPositionNew: otherBotPosition };
    this._layout = layout;
    const ev = layout.event;
    Object.assign(this._episodeMetadata, {
      scene_check_passed: true,
      scene_search_attempts: layout.attempts,
      block: ev.block,
      target_cell: ev.pos,
      pedestal_cell: ev.base_pos,
      pedestal_block: COLUMN_BASE_BLOCK,
      held_item: isActor ? layout.held_item : null,
      actor_held_item: layout.held_item,
      layout: {
        a_pos: layout.posA,
        b_pos: layout.posB,
        a_yaw: layout.yawA,
        b_start_yaw: layout.yawB0,
        b_end_yaw: layout.yawBEnd,
        a_target_distance: layout.distA,
        b_target_distance: layout.distB,
        bot_distance: layout.botDistance,
        a_offset_in_b_end_view_deg: layout.alphaDeg,
        a_offset_in_b_start_view_deg: B_TURN_DEG + layout.alphaDeg,
        b_offset_in_a_view_deg: layout.betaDeg,
        target_offset_in_b_start_view_deg: B_TURN_DEG,
        yaw_convention: "mineflayer: yaw 0 = north = -z, grows counter-clockwise (left)",
      },
      start_pos: isActor ? layout.posA : layout.posB,
      start_yaw: isActor ? layout.yawA : layout.yawB0,
      start_pitch: 0,
      event: ev,
      events: [ev],
    });
    Object.assign(this._evalMetadata, { block: ev.block });

    const [px, py, pz] = isActor ? layout.posA : layout.posB;
    const yaw = isActor ? layout.yawA : layout.yawB0;
    return {
      botPositionNew: new Vec3(px, py, pz),
      otherBotPositionNew: new Vec3(px - Math.sin(yaw) * 16, py, pz - Math.cos(yaw) * 16),
    };
  }

  async actorSetup(bot, rcon, episodeNum, args, botPosition, seed, spec) {
    const rng = seedrandom(seed);
    const block = BLOCK_TYPES[Math.floor(rng() * BLOCK_TYPES.length)];
    const sign = spec.direction === "left" ? 1 : -1;
    console.log(`[${bot.username}] Episode ${episodeNum}: DSP actor, ${spec.interaction} ${block}, H ${spec.hold_frames}, B turns ${spec.direction}`);

    let scene = null;
    let attempts = 0;
    while (scene === null && attempts < SCENE_MAX_ATTEMPTS) {
      if (attempts > 0) {
        const range = Math.min(SCENE_HOP_RANGE * 2 ** (attempts - 1), SCENE_HOP_MAX_RANGE);
        await rconLog(bot, rcon, `spreadplayers ${Math.floor(botPosition.x)} ${Math.floor(botPosition.z)} 0 ${range} false ${bot.username}`);
        await sleep(SCENE_HOP_SETTLE_MS);
      }
      attempts++;
      scene = await pickOffscreenScene(bot, rng, sign);
    }
    this._episodeMetadata.scene_search_attempts = attempts;
    if (scene === null) return null;
    const { posA, posB } = scene;
    const ev = { ...scene.ev, type: spec.interaction, block, actor: bot.username };
    console.log(
      `[${bot.username}] scene: target ${ev.pos}, A ${posA} (${scene.distA.toFixed(2)}), B ${posB} (${scene.distB.toFixed(2)}), ` +
        `alpha ${scene.alphaDeg.toFixed(1)} beta ${scene.betaDeg.toFixed(1)} deg, ${attempts} attempt(s)`,
    );
    for (const [name, pos] of [[bot.username, posA], [args.other_bot_name, posB]]) {
      await rconLog(bot, rcon, `tp ${name} ${pos.x} ${pos.y} ${pos.z}`);
    }
    await sleep(1500);
    if (bot.entity.position.distanceTo(posA) > 0.3) {
      console.log(`[${bot.username}] actor not at the scene start (${bot.entity.position}), giving up`);
      return null;
    }
    await prepareWorld(bot, rcon, posA);
    this._builtEvent = ev;
    if (!(await buildColumn(bot, rcon, ev))) return null;
    await rcon.send(`clear ${args.other_bot_name}`);
    const hands = await armActor(bot, rcon, ev);
    if (!hands) return null;
    return {
      event: ev,
      posA: [posA.x, posA.y, posA.z],
      posB: [posB.x, posB.y, posB.z],
      yawA: scene.yawA,
      yawB0: scene.yawB0,
      yawBEnd: scene.yawBEnd,
      distA: scene.distA,
      distB: scene.distB,
      botDistance: scene.botDistance,
      alphaDeg: scene.alphaDeg,
      betaDeg: scene.betaDeg,
      attempts,
      held_item: hands.held_item,
    };
  }

  async entryPoint(bot, rcon, sharedBotRng, coordinator, iterationID, episodeNum, args) {
    coordinator.onceEvent("worldModOffscreenPhase", episodeNum, getOnOffscreenPhaseFn(bot, rcon, sharedBotRng, coordinator, episodeNum, this, args));
    coordinator.sendToOtherBot("worldModOffscreenPhase", bot.entity.position.clone(), episodeNum, "entryPoint end");
  }

  async tearDownEpisode(bot, rcon, sharedBotRng, coordinator, episodeNum, args) {
    if (!this._isActor) return;
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

module.exports = { WorldModOffscreenEvalEpisode, _internals: { offscreenSpec, offscreenPhasePlan, bTurnDeg } };

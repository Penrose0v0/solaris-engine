const { stopAll } = require("../../primitives/movement");
const { coObserveUtils } = require("./co-observe-eval-episode");
const { SharedArenaEvalEpisode, sharedArenaUtils } = require("./shared-arena-eval-episode");

const { rendezvous, eyeOf, DEG } = coObserveUtils;
const { wallHit, baselineDots, WALL_OFFSET } = sharedArenaUtils;

// ---------------------------------------------------------------------------
// Long-term shared-world consistency, cross revisit. Both bots stand in a
// sharedArenaEval courtyard and never translate; pitch 0 throughout.
//
// Azimuths (north 0, east 90; mineflayer yaw = -azimuth, north = -z,
// east = +x): A stands west, B east, BOT_DISTANCE apart on an east-west
// line. A starts at azimuth 130, B at 230: each has the other
// PARTNER_OFFSET_DEG off its view centre and the north wall ("x") behind
// the edge of its view.
//
// Script in recorded frames (20 fps), "first" = the first observer:
//   static_pre     10   nobody moves
//   first_turn     33   first turns TURN_DEG to north (4 deg per frame, the
//                       last frame 2 deg)
//   first_observe  15   first looks at x
//   first_return   33   first turns back (control: keeps looking at x)
//   gap             G   nobody moves (main group: nobody sees x)
//   second_turn    33   second turns to north
//   second_observe 15   second looks at x
// The bot that is not turning stands still.
//
// Variants by k = (4 * instance id + episode number) % 8:
//   G = GAP_FRAMES[k % 2], control = bit 1 of k, first observer = B / A by
//   bit 2 of k.
//
// The script is driven by the recorder itself: the frame recorder calls
// bot.getLastCameraAction() exactly once per recorded frame, and the hook
// installed here sets the look for that frame before the recorder reads it.
// Step 0 is the bot's first frame at or after the start time T0 both bots
// agreed on; every later frame advances the script by one step (the recorder
// period is 50.3-50.4 ms, so a 50 ms wall-clock grid would lose a step every
// ~150 frames), or by round(gap / 50 ms) steps after a gap >= 75 ms. So
// every recorded frame carries exactly one script step, its `action.camera`
// is the yaw difference to the previous recorded frame, and a frame the
// recorder skips shows up as one frame with a doubled step in both the pose
// and the action. `frame_trace` keeps (hook time, step) per frame.
// ---------------------------------------------------------------------------
const BOT_DISTANCES = [6, 6.5, 7];
const SOUTH_OFFSETS = [2, 3]; // blocks south of the courtyard centre: the whole north wall is inside both north views
const AZIMUTH_A_DEG = 130;
const AZIMUTH_B_DEG = 230;
const TURN_DEG = 130;
const DEG_PER_FRAME = 4;
const PARTNER_OFFSET_DEG = 40;
const STATIC_PRE_FRAMES = 10;
const TURN_FRAMES = Math.ceil(TURN_DEG / DEG_PER_FRAME); // 33
const OBSERVE_FRAMES = 15;
const GAP_FRAMES = [5, 55];
const TAIL_FRAMES = 15;
const FRAME_MS = 50;
const DROP_GAP_MS = 75;
const LEAD_TICKS = 10;
const SYNC_WIGGLE_DEG = 6;
const START_DELAY_MS = 1000;

function crossRevisitSpec(instanceId, episodeNum) {
  const k = (((4 * instanceId + episodeNum) % 8) + 8) % 8;
  return {
    variant_index: k,
    gap_frames: GAP_FRAMES[k % 2],
    is_control: ((k >> 1) & 1) === 1,
    first_observer_agent: ((k >> 2) & 1) === 1 ? "A" : "B",
  };
}

function phasePlan(spec) {
  const G = spec.gap_frames;
  const first = spec.first_observer_agent;
  const second = first === "A" ? "B" : "A";
  const lens = [
    ["static_pre", STATIC_PRE_FRAMES, null],
    ["first_turn", TURN_FRAMES, first],
    ["first_observe", OBSERVE_FRAMES, first],
    ["first_return", TURN_FRAMES, spec.is_control ? null : first],
    ["gap", G, null],
    ["second_turn", TURN_FRAMES, second],
    ["second_observe", OBSERVE_FRAMES, second],
  ];
  let s = 0;
  return lens.map(([name, frames, actor]) => {
    const p = { name, start_step: s, end_step: s + frames - 1, frames, turning_agent: name.endsWith("observe") ? null : actor, observing_agent: null };
    if (name === "first_observe" || (spec.is_control && ["first_return", "gap", "second_turn", "second_observe"].includes(name))) p.observing_agent = first;
    if (name === "second_observe") p.observing_agent = p.observing_agent ? "both" : second;
    s += frames;
    return p;
  });
}

// Degrees turned towards north at script step s.
function turnedDeg(spec, isFirst, s) {
  const t1 = STATIC_PRE_FRAMES; // first_turn start
  const r1 = t1 + TURN_FRAMES + OBSERVE_FRAMES; // first_return start
  const t2 = r1 + TURN_FRAMES + spec.gap_frames; // second_turn start
  if (!isFirst) return s < t2 ? 0 : Math.min(DEG_PER_FRAME * (s - t2 + 1), TURN_DEG);
  if (s < t1) return 0;
  if (s < r1 || spec.is_control) return Math.min(DEG_PER_FRAME * (s - t1 + 1), TURN_DEG);
  return TURN_DEG - Math.min(DEG_PER_FRAME * (s - r1 + 1), TURN_DEG);
}

function getOnCrossRevisitPhaseFn(bot, rcon, sharedBotRng, coordinator, episodeNum, episodeInstance, args) {
  return async () => {
    coordinator.sendToOtherBot("crossRevisitPhase", bot.entity.position.clone(), episodeNum, "crossRevisitPhase beginning");
    const layout = episodeInstance._layout;
    const stopNow = () => {
      coordinator.onceEvent(
        "stopPhase",
        episodeNum,
        episodeInstance.getOnStopPhaseFn(bot, rcon, sharedBotRng, coordinator, args.other_bot_name, episodeNum, args),
      );
      coordinator.sendToOtherBot("stopPhase", bot.entity.position.clone(), episodeNum, "crossRevisitPhase end");
    };
    if (!layout) {
      console.log(`[${bot.username}] no layout (scene setup failed), stopping`);
      stopNow();
      return;
    }
    const spec = episodeInstance._crossSpec;
    const me = layout.bots[bot.username];
    const meta = episodeInstance._episodeMetadata;
    const isFirst = me.agent === spec.first_observer_agent;
    const sign = me.agent === "A" ? 1 : -1; // A turns left (yaw grows), B right
    const windowSteps = STATIC_PRE_FRAMES + 3 * TURN_FRAMES + 2 * OBSERVE_FRAMES + spec.gap_frames;
    const trace = [];
    const start = bot.entity.position.clone();
    const original = bot.getLastCameraAction;
    try {
      stopAll(bot);
      // The recording camera only re-syncs when the bot turned > 1 deg, so
      // it can sit up to 1 deg off after the setup's aim: force a sync.
      await bot.look(me.yaw0 + SYNC_WIGGLE_DEG * DEG, SYNC_WIGGLE_DEG * DEG, true);
      await bot.waitForTicks(4);
      await bot.look(me.yaw0, 0, true);
      await bot.waitForTicks(LEAD_TICKS);

      let t0Promise = null;
      if (!episodeInstance._isPrimary) {
        t0Promise = new Promise((resolve) => coordinator.onceEvent("crossRevisitT0", episodeNum, (data) => resolve(data.t0)));
      }
      await rendezvous(coordinator, "crossRevisitGo", episodeNum);
      let t0;
      if (episodeInstance._isPrimary) {
        t0 = Date.now() + START_DELAY_MS;
        coordinator.sendToOtherBot("crossRevisitT0", { t0 }, episodeNum, "crossRevisitT0 send");
      } else {
        t0 = await t0Promise;
      }

      let anchor = null;
      let lastDeg = 0;
      let step = 0;
      let lastFrameMs = 0;
      let finish;
      const done = new Promise((resolve) => (finish = resolve));
      bot.getLastCameraAction = () => {
        try {
          const now = Date.now();
          if (anchor === null && now >= t0) {
            anchor = now;
            step = 0;
          } else if (anchor !== null) {
            // One step per recorded frame; a frame the recorder really
            // skipped (gap of 1.5 frame periods or more) takes its steps along.
            const dt = now - lastFrameMs;
            step += dt >= DROP_GAP_MS ? Math.round(dt / FRAME_MS) : 1;
          }
          lastFrameMs = now;
          if (anchor !== null) {
            const s = step;
            if (s < windowSteps + TAIL_FRAMES) {
              const deg = turnedDeg(spec, isFirst, Math.min(s, windowSteps - 1));
              if (deg !== lastDeg) {
                lastDeg = deg;
                bot.look(me.yaw0 + sign * deg * DEG, 0, true).catch(() => {});
              }
              trace.push([now, s, deg]);
            } else {
              finish();
            }
          }
        } catch (err) {
          console.log(`[${bot.username}] frame hook error: ${err.message}`);
        }
        return original();
      };
      const timeout = new Promise((resolve) => setTimeout(() => resolve("timeout"), START_DELAY_MS + (windowSteps + TAIL_FRAMES) * FRAME_MS + 15000));
      const how = await Promise.race([done, timeout]);
      bot.getLastCameraAction = original;
      if (how === "timeout") console.log(`[${bot.username}] cross revisit: frame hook timed out after ${trace.length} frames`);

      const inWindow = trace.filter((p) => p[1] < windowSteps);
      meta.t0_time_ms = t0;
      meta.anchor_time_ms = anchor;
      meta.window_start_time_ms = anchor;
      meta.window_end_time_ms = inWindow.length ? inWindow[inWindow.length - 1][0] + FRAME_MS / 2 : null;
      meta.frame_trace = trace;
      meta.frame_trace_format = "[hook time ms (the frame's renderTime is 0-2 ms later), script step, degrees turned towards north]";
      meta.hook_timed_out = how === "timeout";
      const end = bot.entity.position;
      meta.drift_blocks = Math.hypot(end.x - start.x, end.y - start.y, end.z - start.z);
      const steps = inWindow.map((p) => p[1]);
      const skipped = steps.slice(1).filter((s, i) => s - steps[i] !== 1).length;
      console.log(
        `[${bot.username}] cross revisit done: agent ${me.agent} (${isFirst ? "first" : "second"} observer), G ${spec.gap_frames}, ` +
          `control ${spec.is_control}, ${inWindow.length}/${windowSteps} window frames, irregular steps ${skipped}, drift ${meta.drift_blocks.toFixed(4)}`,
      );
    } finally {
      bot.getLastCameraAction = original;
      stopAll(bot);
    }
    stopNow();
  };
}

/**
 * crossRevisitEval: long-term shared-world consistency by cross revisit.
 * In a sharedArenaEval courtyard A (west) and B (east) stand 6-7 blocks
 * apart, each seeing the other 40 deg off centre and neither seeing the
 * north wall x. The first observer turns to x, looks, turns back; after a
 * gap of G frames in which nobody sees x, the second observer turns to x.
 * Control episodes: the first observer keeps looking at x. See the header
 * of this file for the script and the variants.
 *
 * episode_info at record time: `cross_revisit` (roles, G, control, geometry),
 * `phase_plan` (script steps), `frame_trace`. postprocess/cross_revisit_quality.py
 * turns them into `phases` (start_frame / end_frame) and `x_visibility`
 * (GT frustum) in the flattened dataset.
 * @extends SharedArenaEvalEpisode
 */
class CrossRevisitEvalEpisode extends SharedArenaEvalEpisode {
  static SEED_PREFIX = "crossRevisitEval";

  planPlacement(rng, centre) {
    const d = BOT_DISTANCES[Math.floor(rng() * BOT_DISTANCES.length)];
    const south = SOUTH_OFFSETS[Math.floor(rng() * SOUTH_OFFSETS.length)];
    const posA = { x: centre.x - d / 2, y: centre.y, z: centre.z + south };
    const posB = { x: centre.x + d / 2, y: centre.y, z: centre.z + south };
    const yawA = -AZIMUTH_A_DEG * DEG;
    const yawB = (360 - AZIMUTH_B_DEG) * DEG;
    return {
      posA,
      posB,
      yawA,
      yawB,
      hitA: wallHit(centre, eyeOf(posA), yawA),
      hitB: wallHit(centre, eyeOf(posB), yawB),
      dots: baselineDots(posA, posB, yawA, yawB),
      tries: 1,
      sweeps: [
        { agent: "A", pitch0: 0, azimuth0_deg: AZIMUTH_A_DEG, turn: "left" },
        { agent: "B", pitch0: 0, azimuth0_deg: AZIMUTH_B_DEG, turn: "right" },
      ],
    };
  }

  async setupEpisode(bot, rcon, sharedBotRng, coordinator, episodeNum, args, botPosition, otherBotPosition) {
    const spec = crossRevisitSpec(args.instance_id ?? 0, episodeNum);
    this._crossSpec = spec;
    const out = await super.setupEpisode(bot, rcon, sharedBotRng, coordinator, episodeNum, args, botPosition, otherBotPosition);
    const layout = this._layout;
    const meta = this._episodeMetadata;
    meta.lead_ticks = LEAD_TICKS;
    delete meta.motion_ticks;
    delete meta.tail_ticks;
    meta.tail_frames = TAIL_FRAMES;
    Object.assign(this._evalMetadata, { gap_frames: spec.gap_frames, is_control: spec.is_control });
    if (!layout) {
      meta.cross_revisit_planned = spec;
      return out;
    }
    const agents = {};
    for (const b of Object.values(layout.bots)) agents[b.agent] = b.name;
    const first = spec.first_observer_agent;
    const second = first === "A" ? "B" : "A";
    const me = layout.bots[bot.username];
    const cz = layout.centre.z;
    const cx = layout.centre.x;
    delete meta.look_plan;
    Object.assign(meta, {
      agent: me.agent,
      observer_role: me.agent === first ? "first" : "second",
      cross_revisit: {
        ...spec,
        role_order: `${first} looks first, ${second} revisits`,
        agents,
        first_observer: agents[first],
        second_observer: agents[second],
        azimuth_convention: "north 0, east 90 (clockwise); mineflayer yaw = -azimuth; north = -z, east = +x",
        start_azimuth_deg: { A: AZIMUTH_A_DEG, B: AZIMUTH_B_DEG },
        turn: { A: "left", B: "right" },
        turn_deg: TURN_DEG,
        deg_per_frame: DEG_PER_FRAME,
        partner_offset_deg: PARTNER_OFFSET_DEG,
        bot_distance: layout.start_overlap.bot_distance,
        north_wall_distance: WALL_OFFSET - 0.5 + (me.pos[2] - cz),
        x: {
          what: "north wall: the five modules of side 'north' (blocks of courtyard.blocks whose module is one of them)",
          inner_face_z: cz - (WALL_OFFSET - 0.5),
          x_range: [cx - (WALL_OFFSET - 0.5), cx + (WALL_OFFSET - 0.5)],
          modules: layout.courtyard.modules.filter((m) => m.side === "north"),
        },
        window_frames: STATIC_PRE_FRAMES + 3 * TURN_FRAMES + 2 * OBSERVE_FRAMES + spec.gap_frames,
        fps: 20,
      },
      phase_plan: phasePlan(spec),
      suggested_window: "window_start_frame .. window_end_frame (static_pre .. second_observe); frames before it are the static lead, after it a static tail",
    });
    Object.assign(this._evalMetadata, { first_observer: agents[first], second_observer: agents[second] });
    return out;
  }

  async entryPoint(bot, rcon, sharedBotRng, coordinator, iterationID, episodeNum, args) {
    coordinator.onceEvent(
      "crossRevisitPhase",
      episodeNum,
      getOnCrossRevisitPhaseFn(bot, rcon, sharedBotRng, coordinator, episodeNum, this, args),
    );
    coordinator.sendToOtherBot("crossRevisitPhase", bot.entity.position.clone(), episodeNum, "entryPoint end");
  }
}

module.exports = { CrossRevisitEvalEpisode, _internals: { crossRevisitSpec, phasePlan, turnedDeg } };

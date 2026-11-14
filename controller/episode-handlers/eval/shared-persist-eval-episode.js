const { coObserveUtils } = require("./co-observe-eval-episode");
const { sharedArenaUtils } = require("./shared-arena-eval-episode");
const { CrossRevisitEvalEpisode, crossRevisitConstants } = require("./cross-revisit-eval-episode");

const { eyeOf, DEG } = coObserveUtils;
const { wallHit, baselineDots, WALL_OFFSET } = sharedArenaUtils;
const { STATIC_PRE_FRAMES, TURN_FRAMES, OBSERVE_FRAMES, GAP_FRAMES, TURN_DEG, DEG_PER_FRAME, PARTNER_OFFSET_DEG } = crossRevisitConstants;

// ---------------------------------------------------------------------------
// Shared-world persistence with a conditioning frame. Courtyard, standing
// cells and the frame-driven script of crossRevisitEval (A west, B east,
// 6-7 blocks apart on an east-west line, 2-3 blocks south of the centre;
// nobody translates; pitch 0); x is the north wall.
//
// Azimuths: north 0, east 90. "first" = the bot that sees x in the
// conditioning frame (the first frame of the window), "second" = the bot
// that sees x at the end.
//
//   B first (both turn LEFT):  B starts at 0 (north: sees x, not A),
//                              A starts at 130 (sees B 40 deg off centre, not x);
//                              B turns to 230, A turns to 0.
//   A first (both turn RIGHT): A starts at 0, B starts at 230 (sees A, not x);
//                              A turns to 130, B turns to 360.
//
// Script in recorded frames (20 fps):
//   static_pre     10   nobody moves (frame 0 of it = the conditioning frame)
//   first_leave    33   first turns 130 deg away from x (4 deg per frame, the
//                       last frame 2 deg)
//   gap             G   nobody moves, nobody sees x
//   second_turn    33   second turns 130 deg to north
//   second_observe 15   second looks at x
//
// Variants by k = (2 * instance id + episode number) % 4:
//   G = GAP_FRAMES[k % 2], first = B / A by bit 1 of k.
// ---------------------------------------------------------------------------
const AZIMUTH_NORTH_DEG = 0;
const AZIMUTH_A_AWAY_DEG = 130; // A when it does not look at x
const AZIMUTH_B_AWAY_DEG = 230; // B when it does not look at x

function sharedPersistSpec(instanceId, episodeNum) {
  const k = (((2 * instanceId + episodeNum) % 4) + 4) % 4;
  const first = ((k >> 1) & 1) === 1 ? "A" : "B";
  return {
    variant_index: k,
    gap_frames: GAP_FRAMES[k % 2],
    first_observer_agent: first,
    turn_direction: first === "B" ? "left" : "right",
  };
}

function phasePlan(spec) {
  const first = spec.first_observer_agent;
  const second = first === "A" ? "B" : "A";
  const lens = [
    ["static_pre", STATIC_PRE_FRAMES, null, first],
    ["first_leave", TURN_FRAMES, first, null],
    ["gap", spec.gap_frames, null, null],
    ["second_turn", TURN_FRAMES, second, null],
    ["second_observe", OBSERVE_FRAMES, null, second],
  ];
  let s = 0;
  return lens.map(([name, frames, turning, observing]) => {
    const p = { name, start_step: s, end_step: s + frames - 1, frames, turning_agent: turning, observing_agent: observing };
    s += frames;
    return p;
  });
}

// Degrees turned away from the start yaw at script step s.
function turnedDeg(spec, isFirst, s) {
  const t0 = isFirst ? STATIC_PRE_FRAMES : STATIC_PRE_FRAMES + TURN_FRAMES + spec.gap_frames;
  return s < t0 ? 0 : Math.min(DEG_PER_FRAME * (s - t0 + 1), TURN_DEG);
}

/**
 * sharedPersistEval: shared-world persistence with a conditioning frame.
 * In the crossRevisitEval courtyard the first bot looks at the north wall x
 * from the start (the other sees the first bot but not x), turns away; after
 * a gap of G frames in which nobody sees x, the other bot turns to x. See
 * the header of this file.
 *
 * episode_info at record time: `shared_persist` (roles, G, geometry),
 * `phase_plan`, `frame_trace`. postprocess/shared_persist_quality.py adds
 * `phases`, `condition_frame` and `x_visibility` in the flattened dataset.
 * @extends CrossRevisitEvalEpisode
 */
class SharedPersistEvalEpisode extends CrossRevisitEvalEpisode {
  static SEED_PREFIX = "sharedPersistEval";

  makeSpec(instanceId, episodeNum) {
    return sharedPersistSpec(instanceId, episodeNum);
  }

  windowSteps(spec) {
    return STATIC_PRE_FRAMES + 2 * TURN_FRAMES + OBSERVE_FRAMES + spec.gap_frames;
  }

  yawOffsetDeg(spec, me, s) {
    const sign = spec.turn_direction === "left" ? 1 : -1; // left = yaw grows
    return sign * turnedDeg(spec, me.agent === spec.first_observer_agent, s);
  }

  startAzimuths(spec) {
    return spec.first_observer_agent === "B"
      ? { A: AZIMUTH_A_AWAY_DEG, B: AZIMUTH_NORTH_DEG }
      : { A: AZIMUTH_NORTH_DEG, B: AZIMUTH_B_AWAY_DEG };
  }

  planPlacement(rng, centre) {
    const spec = this._crossSpec;
    const { posA, posB } = this.planPositions(rng, centre);
    const az = this.startAzimuths(spec);
    const yawA = -az.A * DEG;
    const yawB = az.B === 0 ? 0 : (360 - az.B) * DEG;
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
        { agent: "A", pitch0: 0, azimuth0_deg: az.A, turn: spec.turn_direction },
        { agent: "B", pitch0: 0, azimuth0_deg: az.B, turn: spec.turn_direction },
      ],
    };
  }

  describeEpisode(bot, spec, layout, meta) {
    if (!layout) {
      meta.shared_persist_planned = spec;
      return;
    }
    const agents = {};
    for (const b of Object.values(layout.bots)) agents[b.agent] = b.name;
    const first = spec.first_observer_agent;
    const second = first === "A" ? "B" : "A";
    const me = layout.bots[bot.username];
    const start = this.startAzimuths(spec);
    const end = first === "B" ? { A: AZIMUTH_NORTH_DEG, B: AZIMUTH_B_AWAY_DEG } : { A: AZIMUTH_A_AWAY_DEG, B: AZIMUTH_NORTH_DEG };
    Object.assign(meta, {
      agent: me.agent,
      observer_role: me.agent === first ? "first" : "second",
      shared_persist: {
        ...spec,
        role_order: `${first} sees x in the conditioning frame and turns away, ${second} turns to x`,
        agents,
        first_observer: agents[first],
        second_observer: agents[second],
        azimuth_convention: "north 0, east 90 (clockwise); mineflayer yaw = -azimuth; north = -z, east = +x",
        start_azimuth_deg: start,
        end_azimuth_deg: end,
        turn: { A: spec.turn_direction, B: spec.turn_direction },
        turn_deg: TURN_DEG,
        deg_per_frame: DEG_PER_FRAME,
        partner_offset_deg: PARTNER_OFFSET_DEG,
        bot_distance: layout.start_overlap.bot_distance,
        north_wall_distance: WALL_OFFSET - 0.5 + (me.pos[2] - layout.centre.z),
        x: this.describeX(layout),
        window_frames: this.windowSteps(spec),
        fps: 20,
      },
      phase_plan: phasePlan(spec),
      suggested_window:
        "window_start_frame .. window_end_frame (static_pre .. second_observe); window_start_frame is the conditioning frame; " +
        "frames before it are the static lead (it contains a short camera sync wiggle), after it a static tail",
    });
    Object.assign(this._evalMetadata, { first_observer: agents[first], second_observer: agents[second] });
  }
}

module.exports = { SharedPersistEvalEpisode, _internals: { sharedPersistSpec, phasePlan, turnedDeg } };

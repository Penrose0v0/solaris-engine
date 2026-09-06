const { sneak, stopAll } = require("../../primitives/movement");
const { BaseEpisode } = require("../base-episode");

// Recording ~20 fps. Safety cap on wall-clock; actual duration is
// arc_degrees / omega_deg_per_sec.
const EPISODE_MAX_TICKS = 300;

// Poll interval for the yaw-update loop. 20 fps = 50ms/tick; a 1-tick poll
// gives us a yaw update every recording frame.
const ORBIT_POLL_TICKS = 1;

// Nominal MC walk speed (blocks/sec). Sprint intentionally NOT used here
// because sprint changes tangential speed and thus orbit radius; keeping
// walk-only gives a single deterministic radius per omega.
const NOMINAL_WALK_SPEED_BLOCKS_PER_SEC = 4.317;

// Parameter grid. arc_degrees is signed (sign = orbit direction; positive =
// CCW as viewed from above in MC yaw convention). |arc| always <= 180 to
// keep episode wall-clock manageable at the slower omega values.
//
// Radius r = v / omega (radians/sec):
//   omega =  30 deg/s = 0.524 rad/s -> r = 4.317 / 0.524 = 8.24 blocks
//   omega =  60 deg/s = 1.047 rad/s -> r = 4.317 / 1.047 = 4.12 blocks
// Wall-clock:
//   90 deg / 30 deg/s = 3.0s;   90 / 60 = 1.5s
//   180 deg / 30 = 6.0s;       180 / 60 = 3.0s
const ARC_DEGREES_GRID = [90, -90, 180, -180];
const OMEGA_DEG_PER_SEC_GRID = [30, 60];

// Which bot performs the orbit (the other stands still as a scene element).
function pickOrbitBot(bot, otherBotName, episodeNum) {
  const [lower, higher] =
    bot.username < otherBotName
      ? [bot.username, otherBotName]
      : [otherBotName, bot.username];
  return episodeNum % 2 === 0 ? lower : higher;
}

function pickTrajectoryParams(episodeNum) {
  const cols = OMEGA_DEG_PER_SEC_GRID.length;
  const rows = ARC_DEGREES_GRID.length;
  const idx = episodeNum % (rows * cols);
  const arcDegrees = ARC_DEGREES_GRID[Math.floor(idx / cols)];
  const omega = OMEGA_DEG_PER_SEC_GRID[idx % cols];
  return { arcDegrees, omega };
}

function getOnOrbitPhaseFn(
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
      "orbitPhase",
      bot.entity.position.clone(),
      episodeNum,
      "orbitPhase beginning",
    );

    const orbitBot = pickOrbitBot(bot, args.other_bot_name, episodeNum);
    const { arcDegrees, omega } = pickTrajectoryParams(episodeNum);
    const shouldThisBotOrbit = bot.username === orbitBot;

    // r = v / omega; radius characterizes the traced circle.
    const omegaRad = (omega * Math.PI) / 180;
    const nominalRadius = NOMINAL_WALK_SPEED_BLOCKS_PER_SEC / omegaRad;

    episodeInstance._episodeMetadata = {
      traj_kind: "orbit",
      sweep_bot: orbitBot,
      sweep_degrees: arcDegrees,
      omega_deg_per_sec: omega,
      nominal_walk_speed_blocks_per_sec: NOMINAL_WALK_SPEED_BLOCKS_PER_SEC,
      nominal_radius_blocks: nominalRadius,
      traj_expected_ticks: Math.round((Math.abs(arcDegrees) / omega) * 20),
    };
    episodeInstance._evalMetadata = {
      bots_chosen: [orbitBot],
      sweep_degrees: arcDegrees,
      camera_speed_degrees_per_sec: omega,
      nominal_radius_blocks: nominalRadius,
    };

    console.log(
      `[${bot.username}] Episode ${episodeNum}: orbit_bot=${orbitBot}, ` +
        `arc=${arcDegrees} deg @ ${omega} deg/s -> r=${nominalRadius.toFixed(2)} blocks, ` +
        `this bot ${shouldThisBotOrbit ? "ORBITS" : "stays still"}`,
    );

    if (shouldThisBotOrbit) {
      // Sneak-signal so downstream can locate the "orbit starts here" frame.
      await sneak(bot);

      const startYaw = bot.entity.yaw;
      const startPitch = bot.entity.pitch;
      const startTick = bot.time.age;
      // Signed step per poll: |arcDegrees| covered over |arcDegrees|/omega
      // seconds = |arcDegrees|/omega * 20 ticks. Per ORBIT_POLL_TICKS step
      // that is (omega * ORBIT_POLL_TICKS / 20) degrees, signed by arc dir.
      const arcSign = Math.sign(arcDegrees);
      const degPerPoll = arcSign * omega * (ORBIT_POLL_TICKS / 20);
      const radPerPoll = (degPerPoll * Math.PI) / 180;
      const totalRadTarget = (arcDegrees * Math.PI) / 180;

      // Start walking; the walk direction is the current yaw. As we rotate
      // yaw, the walk direction rotates with it, tracing a circle.
      bot.setControlState("forward", true);

      let accumulatedRad = 0;
      while (true) {
        const remainingRad = totalRadTarget - accumulatedRad;
        // Are we done (with a small epsilon)?
        if (Math.abs(remainingRad) < 1e-4) break;
        const elapsedTicks = bot.time.age - startTick;
        if (elapsedTicks >= EPISODE_MAX_TICKS) break;

        const step =
          Math.abs(remainingRad) < Math.abs(radPerPoll)
            ? remainingRad
            : radPerPoll;
        accumulatedRad += step;
        const nextYaw = startYaw + accumulatedRad;
        // Fast look() so the yaw update is instantaneous (~1 tick):
        // we want the yaw to change strictly monotonically at rate omega,
        // not smoothly interpolated.
        await bot.look(nextYaw, startPitch, true);
        await bot.waitForTicks(ORBIT_POLL_TICKS);
      }

      stopAll(bot);
      // Let physics settle so trailing frames are stationary.
      await bot.waitForTicks(20);
    } else {
      // Non-orbit bot: stand still for the full max duration.
      await bot.waitForTicks(EPISODE_MAX_TICKS);
    }

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
      "orbitPhase end",
    );
  };
}

/**
 * Atomic eval episode: one bot walks forward at MC's nominal walk speed while
 * its yaw rotates at a fixed angular rate. Because the walk direction is
 * always the current yaw, the bot traces a circle of radius r = v / omega
 * centered on a point perpendicular to the initial facing. The other bot
 * stands still as a scene landmark.
 *
 * Per-frame ground truth is parametric:
 *   theta(t) = start_yaw + omega * t
 *   x(t)     = start_x + r * (cos(start_yaw - PI/2) - cos(theta(t) - PI/2))
 *   z(t)     = start_z + r * (sin(start_yaw - PI/2) - sin(theta(t) - PI/2))
 *   yaw(t)   = start_yaw + omega * t   (camera pans with translation)
 *
 * Serves as a rotation+translation coupled baseline complementing the pure-
 * rotation (yaw-sweep, pitch-sweep), pure-translation (forward), and
 * zero-motion (static) atomic primitives.
 * @extends BaseEpisode
 */
class TrajOrbitEvalEpisode extends BaseEpisode {
  static WORKS_IN_NON_FLAT_WORLD = true;
  static INIT_MIN_BOTS_DISTANCE = 16;
  static INIT_MAX_BOTS_DISTANCE = 24;

  async entryPoint(
    bot,
    rcon,
    sharedBotRng,
    coordinator,
    iterationID,
    episodeNum,
    args,
  ) {
    coordinator.onceEvent(
      "orbitPhase",
      episodeNum,
      getOnOrbitPhaseFn(
        bot,
        rcon,
        sharedBotRng,
        coordinator,
        episodeNum,
        this,
        args,
      ),
    );
    coordinator.sendToOtherBot(
      "orbitPhase",
      bot.entity.position.clone(),
      episodeNum,
      "teleportPhase end",
    );
  }
}

module.exports = { TrajOrbitEvalEpisode };

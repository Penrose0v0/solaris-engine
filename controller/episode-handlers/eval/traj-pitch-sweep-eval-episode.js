const { sneak, lookSmooth } = require("../../primitives/movement");
const { BaseEpisode } = require("../base-episode");

// mineflayer physics ceiling; see prismarine-physics constants.
const MAX_CAMERA_SPEED_DEGREES_PER_SEC = 171.8873;

// Minimum episode length in ticks so downstream frame-count checks pass
// (recording ~20 fps, 300 ticks ~= 15s of settled + swept camera).
const EPISODE_MIN_TICKS = 300;

// MC clamps pitch to [-90, 90] degrees. Keep a small margin so the requested
// sweep magnitude equals the achieved magnitude (no silent clipping at the pole).
const PITCH_LIMIT_RAD = (89 * Math.PI) / 180;

// Parameter grid. episodeNum indexes into (sweepDegrees x omega) row-major.
// Signed sweeps so we cover both directions per magnitude. Max magnitude 60
// keeps the sweep well inside the [-90, 90] pitch cap even when the initial
// look is not perfectly horizontal (we re-normalize to 0 before sweeping).
const SWEEP_DEGREES_GRID = [15, -15, 30, -30, 45, -45, 60, -60];
const OMEGA_DEG_PER_SEC_GRID = [30, 60, 120];

// Which bot performs the sweep (the other stands still as a scene element).
// Alternated by episodeNum so both bots' recordings are used as the moving
// camera roughly equally.
function pickSweepBot(bot, otherBotName, episodeNum) {
  const [lower, higher] =
    bot.username < otherBotName
      ? [bot.username, otherBotName]
      : [otherBotName, bot.username];
  return episodeNum % 2 === 0 ? lower : higher;
}

function pickTrajectoryParams(episodeNum) {
  const cols = OMEGA_DEG_PER_SEC_GRID.length;
  const rows = SWEEP_DEGREES_GRID.length;
  const idx = episodeNum % (rows * cols);
  const sweepDegrees = SWEEP_DEGREES_GRID[Math.floor(idx / cols)];
  let omega = OMEGA_DEG_PER_SEC_GRID[idx % cols];
  if (omega > MAX_CAMERA_SPEED_DEGREES_PER_SEC) {
    omega = MAX_CAMERA_SPEED_DEGREES_PER_SEC;
  }
  return { sweepDegrees, omega };
}

function clampPitch(pitch) {
  return Math.max(-PITCH_LIMIT_RAD, Math.min(PITCH_LIMIT_RAD, pitch));
}

function getOnPitchSweepPhaseFn(
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
      "pitchSweepPhase",
      bot.entity.position.clone(),
      episodeNum,
      "pitchSweepPhase beginning",
    );

    const sweepBot = pickSweepBot(bot, args.other_bot_name, episodeNum);
    const { sweepDegrees, omega } = pickTrajectoryParams(episodeNum);
    const shouldThisBotSweep = bot.username === sweepBot;

    episodeInstance._episodeMetadata = {
      traj_kind: "pitch-sweep",
      sweep_bot: sweepBot,
      sweep_degrees: sweepDegrees,
      omega_deg_per_sec: omega,
      traj_expected_ticks: EPISODE_MIN_TICKS,
    };
    episodeInstance._evalMetadata = {
      bots_chosen: [sweepBot],
      sweep_degrees: sweepDegrees,
      camera_speed_degrees_per_sec: omega,
    };

    console.log(
      `[${bot.username}] Episode ${episodeNum}: sweep_bot=${sweepBot}, ` +
        `pitch=${sweepDegrees} deg @ ${omega} deg/s, ` +
        `this bot ${shouldThisBotSweep ? "SWEEPS" : "stays still"}`,
    );

    if (shouldThisBotSweep) {
      // Sneak-signal so downstream can locate the "sweep starts here" frame.
      await sneak(bot);

      // Normalize to horizon before sweeping so achieved magnitude matches
      // sweep_degrees exactly (no silent clip at MC's ±90 deg pitch cap).
      const originalYaw = bot.entity.yaw;
      await lookSmooth(bot, originalYaw, 0, MAX_CAMERA_SPEED_DEGREES_PER_SEC, {
        randomized: false,
        useEasing: false,
      });

      const startTick = bot.time.age;
      const targetPitch = clampPitch((sweepDegrees * Math.PI) / 180);

      await lookSmooth(bot, originalYaw, targetPitch, omega, {
        randomized: false,
        useEasing: false,
      });

      const endTick = bot.time.age;
      const remainingTicks = EPISODE_MIN_TICKS - (endTick - startTick);
      if (remainingTicks > 0) {
        await bot.waitForTicks(remainingTicks);
      }
    } else {
      // Non-sweep bot: stand still for the full episode duration so both
      // recordings run for the same time.
      await bot.waitForTicks(EPISODE_MIN_TICKS);
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
      "pitchSweepPhase end",
    );
  };
}

/**
 * Atomic eval episode: one bot stands stationary and sweeps pitch by a fixed
 * signed angle at a fixed angular speed. Camera is first normalized to the
 * horizon (pitch=0) so achieved sweep magnitude equals sweep_degrees exactly.
 * The other bot stands still as a passive scene landmark. Per-frame pitch
 * ground truth is derivable from (0, omega, elapsed_time) once the sweep
 * begins.
 * @extends BaseEpisode
 */
class TrajPitchSweepEvalEpisode extends BaseEpisode {
  static WORKS_IN_NON_FLAT_WORLD = true;
  static INIT_MIN_BOTS_DISTANCE = 8;
  static INIT_MAX_BOTS_DISTANCE = 12;

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
      "pitchSweepPhase",
      episodeNum,
      getOnPitchSweepPhaseFn(
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
      "pitchSweepPhase",
      bot.entity.position.clone(),
      episodeNum,
      "teleportPhase end",
    );
  }
}

module.exports = { TrajPitchSweepEvalEpisode };

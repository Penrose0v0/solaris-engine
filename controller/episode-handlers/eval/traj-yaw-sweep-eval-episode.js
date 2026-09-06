const { sneak, lookSmooth } = require("../../primitives/movement");
const { BaseEpisode } = require("../base-episode");

// mineflayer physics ceiling; see prismarine-physics constants.
const MAX_CAMERA_SPEED_DEGREES_PER_SEC = 171.8873;

// Minimum episode length in ticks so downstream frame-count checks pass
// (recording ~20 fps, 300 ticks ~= 15s of settled + swept camera).
const EPISODE_MIN_TICKS = 300;

// Parameter grid. episodeNum indexes into (sweepDegrees x omega) row-major.
// Signed sweeps so we cover both directions per magnitude.
const SWEEP_DEGREES_GRID = [30, -30, 60, -60, 90, -90, 180, -180];
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

function getOnYawSweepPhaseFn(
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
      "yawSweepPhase",
      bot.entity.position.clone(),
      episodeNum,
      "yawSweepPhase beginning",
    );

    const sweepBot = pickSweepBot(bot, args.other_bot_name, episodeNum);
    const { sweepDegrees, omega } = pickTrajectoryParams(episodeNum);
    const shouldThisBotSweep = bot.username === sweepBot;

    // Metadata is set by both bots so both per-bot episode_info.json files
    // carry the trajectory params; only sweep_bot lets downstream tell which
    // recording is the moving camera.
    episodeInstance._episodeMetadata = {
      traj_kind: "yaw-sweep",
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
        `sweep=${sweepDegrees} deg @ ${omega} deg/s, ` +
        `this bot ${shouldThisBotSweep ? "SWEEPS" : "stays still"}`,
    );

    if (shouldThisBotSweep) {
      // Sneak-signal so downstream can locate the "sweep starts here" frame.
      await sneak(bot);
      const startTick = bot.time.age;

      const originalYaw = bot.entity.yaw;
      const originalPitch = bot.entity.pitch;
      const targetYaw = originalYaw + (sweepDegrees * Math.PI) / 180;

      await lookSmooth(bot, targetYaw, originalPitch, omega, {
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
      "yawSweepPhase end",
    );
  };
}

/**
 * Atomic eval episode: one bot stands stationary and sweeps yaw by a fixed
 * signed angle at a fixed angular speed. The other bot stands still as a
 * passive scene landmark. The trajectory is fully parametrized (sweep_degrees,
 * omega_deg_per_sec) so per-frame yaw ground truth is derivable from
 * (initial_yaw, omega, elapsed_time) even without reading the pose stream.
 * @extends BaseEpisode
 */
class TrajYawSweepEvalEpisode extends BaseEpisode {
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
      "yawSweepPhase",
      episodeNum,
      getOnYawSweepPhaseFn(
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
      "yawSweepPhase",
      bot.entity.position.clone(),
      episodeNum,
      "teleportPhase end",
    );
  }
}

module.exports = { TrajYawSweepEvalEpisode };

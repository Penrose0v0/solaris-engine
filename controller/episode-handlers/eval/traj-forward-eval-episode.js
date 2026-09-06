const { sneak, lookSmooth, stopAll } = require("../../primitives/movement");
const { BaseEpisode } = require("../base-episode");

// mineflayer physics ceiling; see prismarine-physics constants.
const MAX_CAMERA_SPEED_DEGREES_PER_SEC = 171.8873;

// Recording ~20 fps. 300 ticks = 15s wall-clock cap; forward drive terminates
// on distance-reached, so this is a safety ceiling not the target duration.
const EPISODE_MAX_TICKS = 300;

// Poll interval for distance-reached check while driving forward.
const DRIVE_POLL_TICKS = 2;

// Parameter grid. episodeNum indexes into (distance x sprint) row-major.
// Distance is in blocks. Sprint doubles effective speed:
//   walk  ~= 4.317 blocks/s -> 4 blocks ~= 0.93s, 12 blocks ~= 2.78s
//   sprint ~= 5.612 blocks/s -> 4 blocks ~= 0.71s, 12 blocks ~= 2.14s
// Keeping max distance small (12) so the bot stays inside the smoke-test world
// and does not run into unloaded chunks.
const DISTANCE_BLOCKS_GRID = [4, 8, 12];
const SPRINT_GRID = [false, true];

// Which bot performs the drive (the other stands still as a scene element).
// Alternated by episodeNum so both bots' recordings are used as the moving
// camera roughly equally.
function pickDriveBot(bot, otherBotName, episodeNum) {
  const [lower, higher] =
    bot.username < otherBotName
      ? [bot.username, otherBotName]
      : [otherBotName, bot.username];
  return episodeNum % 2 === 0 ? lower : higher;
}

function pickTrajectoryParams(episodeNum) {
  const cols = SPRINT_GRID.length;
  const rows = DISTANCE_BLOCKS_GRID.length;
  const idx = episodeNum % (rows * cols);
  const distanceBlocks = DISTANCE_BLOCKS_GRID[Math.floor(idx / cols)];
  const sprint = SPRINT_GRID[idx % cols];
  return { distanceBlocks, sprint };
}

function horizontalDistanceSq(a, b) {
  const dx = a.x - b.x;
  const dz = a.z - b.z;
  return dx * dx + dz * dz;
}

function getOnForwardDrivePhaseFn(
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
      "forwardDrivePhase",
      bot.entity.position.clone(),
      episodeNum,
      "forwardDrivePhase beginning",
    );

    const driveBot = pickDriveBot(bot, args.other_bot_name, episodeNum);
    const { distanceBlocks, sprint } = pickTrajectoryParams(episodeNum);
    const shouldThisBotDrive = bot.username === driveBot;
    const nominalSpeed = sprint ? 5.612 : 4.317; // blocks/sec, MC constants
    const expectedSeconds = distanceBlocks / nominalSpeed;

    episodeInstance._episodeMetadata = {
      traj_kind: "forward",
      sweep_bot: driveBot,
      sweep_degrees: 0,
      omega_deg_per_sec: 0,
      distance_blocks: distanceBlocks,
      sprint: sprint,
      nominal_speed_blocks_per_sec: nominalSpeed,
      traj_expected_ticks: Math.round(expectedSeconds * 20),
    };
    episodeInstance._evalMetadata = {
      bots_chosen: [driveBot],
      sweep_degrees: 0,
      camera_speed_degrees_per_sec: 0,
      distance_blocks: distanceBlocks,
      sprint: sprint,
    };

    console.log(
      `[${bot.username}] Episode ${episodeNum}: drive_bot=${driveBot}, ` +
        `distance=${distanceBlocks} blocks, sprint=${sprint}, ` +
        `this bot ${shouldThisBotDrive ? "DRIVES FORWARD" : "stays still"}`,
    );

    if (shouldThisBotDrive) {
      // Sneak-signal so downstream can locate the "drive starts here" frame.
      await sneak(bot);

      // Normalize pitch to horizon so the forward vector is horizontal and the
      // resulting motion is pure horizontal translation (no vertical drift).
      const originalYaw = bot.entity.yaw;
      await lookSmooth(bot, originalYaw, 0, MAX_CAMERA_SPEED_DEGREES_PER_SEC, {
        randomized: false,
        useEasing: false,
      });

      const startPos = bot.entity.position.clone();
      const targetDistSq = distanceBlocks * distanceBlocks;
      const startTick = bot.time.age;

      bot.setControlState("forward", true);
      if (sprint) {
        bot.setControlState("sprint", true);
      }

      // Poll until distance-reached or safety-cap on ticks.
      while (true) {
        await bot.waitForTicks(DRIVE_POLL_TICKS);
        const elapsedTicks = bot.time.age - startTick;
        const dSq = horizontalDistanceSq(bot.entity.position, startPos);
        if (dSq >= targetDistSq) break;
        if (elapsedTicks >= EPISODE_MAX_TICKS) break;
      }

      stopAll(bot);

      // Let physics settle so final frames are stationary.
      await bot.waitForTicks(20);
    } else {
      // Non-drive bot: stand still for the full max-episode duration so both
      // recordings run for at least the same wall-clock window.
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
      "forwardDrivePhase end",
    );
  };
}

/**
 * Atomic eval episode: one bot walks (or sprints) straight forward along its
 * current yaw for a fixed distance; the other stands still as a passive scene
 * landmark. Pitch is normalized to the horizon before driving so the motion is
 * pure horizontal translation with no yaw/pitch rotation. Per-frame position
 * ground truth is derivable from (start_pos, yaw, nominal_speed, elapsed_time)
 * once the drive begins.
 * @extends BaseEpisode
 */
class TrajForwardEvalEpisode extends BaseEpisode {
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
      "forwardDrivePhase",
      episodeNum,
      getOnForwardDrivePhaseFn(
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
      "forwardDrivePhase",
      bot.entity.position.clone(),
      episodeNum,
      "teleportPhase end",
    );
  }
}

module.exports = { TrajForwardEvalEpisode };

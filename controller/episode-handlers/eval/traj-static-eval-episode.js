const { sneak } = require("../../primitives/movement");
const { BaseEpisode } = require("../base-episode");

// Minimum episode length in ticks so downstream frame-count checks pass
// (recording ~20 fps, 300 ticks ~= 15s of settled camera).
const EPISODE_MIN_TICKS = 300;

function getOnStaticPhaseFn(
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
      "staticPhase",
      bot.entity.position.clone(),
      episodeNum,
      "staticPhase beginning",
    );

    // Both bots participate as "sweep" here; metadata records both so
    // downstream treats each recording as the moving camera (well, moving
    // nothing) uniformly.
    episodeInstance._episodeMetadata = {
      traj_kind: "static",
      sweep_bot: bot.username,
      sweep_degrees: 0,
      omega_deg_per_sec: 0,
      traj_expected_ticks: EPISODE_MIN_TICKS,
    };
    episodeInstance._evalMetadata = {
      bots_chosen: [bot.username, args.other_bot_name],
      sweep_degrees: 0,
      camera_speed_degrees_per_sec: 0,
    };

    console.log(
      `[${bot.username}] Episode ${episodeNum}: STATIC, both bots stand still ` +
        `for ${EPISODE_MIN_TICKS} ticks`,
    );

    // Sneak-signal so downstream can locate the "episode starts here" frame.
    await sneak(bot);
    await bot.waitForTicks(EPISODE_MIN_TICKS);

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
      "staticPhase end",
    );
  };
}

/**
 * Atomic eval episode: both bots stand completely still for the full episode
 * duration. Camera does not translate or rotate. Serves as a sanity-check
 * baseline: dense-correspondence tools (DA3/VGGT/MapAnything) should score
 * trivially high on pure-static input; a low score here indicates the
 * measurement pipeline itself is broken, not the trajectory design.
 * @extends BaseEpisode
 */
class TrajStaticEvalEpisode extends BaseEpisode {
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
      "staticPhase",
      episodeNum,
      getOnStaticPhaseFn(
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
      "staticPhase",
      bot.entity.position.clone(),
      episodeNum,
      "teleportPhase end",
    );
  }
}

module.exports = { TrajStaticEvalEpisode };

const Vec3 = require("vec3").Vec3;

const { GoalNearXZ } = require("../utils/bot-factory");
const {
  lookAtBot,
  sleep,
  initializePathfinder,
  stopPathfinder,
  land_pos,
} = require("../primitives/movement");
const { BaseEpisode } = require("./base-episode");

// Constants for orbit behavior
const NUM_CHECKPOINTS = 8; // Number of checkpoints around the circle
const CHECKPOINT_REACH_DISTANCE = 1.5; // How close (horizontally) to get to checkpoint (blocks)
const CHECKPOINT_REACH_SLACK = 1.0; // Extra tolerance for pathfinder's block-rounded arrival
const CHECKPOINT_TIMEOUT_MS = 5000; // Maximum time to reach each checkpoint (5 seconds)
const CHECKPOINT_IDLE_GRACE_MS = 800; // Let the pathfinder compute a path before judging idleness
const CHECKPOINT_IDLE_CHECKS = 5; // 5 x 100ms idle => treat as arrived / gave up
// "Moving" but not getting anywhere (digging through / pushing against
// terrain): skip ahead rather than stand still until the timeout.
const CHECKPOINT_STUCK_WINDOW_MS = 1500;
const CHECKPOINT_STUCK_MIN_PROGRESS = 0.3; // blocks of XZ progress expected per window
// Glance at the partner on arrival, but never stand still longer than this:
// the pathfinder overrides look direction every tick while moving, so the
// glance can only happen while stopped, and long stops starve downstream
// camera-translation measurement.
const CHECKPOINT_GLANCE_MAX_MS = 500;
const CAMERA_SPEED_DEGREES_PER_SEC = 171; // Fast glance so it fits in the budget

/**
 * Calculate checkpoint positions around a circle
 * @param {Vec3} center - Center point of the circle
 * @param {number} radius - Radius of the circle
 * @param {number} numCheckpoints - Number of checkpoints to generate
 * @param {number} startAngle - Starting angle in radians (for bot's initial position)
 * @returns {Array<Vec3>} Array of checkpoint positions
 */
function calculateOrbitCheckpoints(
  center,
  radius,
  numCheckpoints,
  startAngle = 0,
) {
  const checkpoints = [];
  const angleStep = (2 * Math.PI) / numCheckpoints;

  for (let i = 0; i < numCheckpoints; i++) {
    const angle = startAngle + i * angleStep;
    const x = center.x + radius * Math.cos(angle);
    const z = center.z + radius * Math.sin(angle);
    checkpoints.push(new Vec3(x, center.y, z));
  }

  return checkpoints;
}

/**
 * Execute orbit by traveling to checkpoints in sequence
 * @param {*} bot - Mineflayer bot instance
 * @param {string} otherBotName - Name of the other bot
 * @param {Array<Vec3>} checkpoints - Array of checkpoint positions
 * @param {Object} rcon - RCON connection for chunk loading
 */
async function executeOrbitWithCheckpoints(
  bot,
  otherBotName,
  checkpoints,
  rcon,
) {
  console.log(
    `[${bot.username}] Starting orbit with ${checkpoints.length} checkpoints`,
  );

  // Initialize pathfinder with full capabilities
  initializePathfinder(bot, {
    allowSprinting: true, // Sprint for faster movement between checkpoints
    allowParkour: true,
    canDig: true,
    canPlaceOn: true,
    allowEntityDetection: true,
  });

  for (let i = 0; i < checkpoints.length; i++) {
    const checkpoint = checkpoints[i];

    // Get ground Y coordinate for checkpoint (land_pos is sync and takes
    // (bot, x, z); on unloaded chunks it returns null -> keep circle Y).
    const groundPos = land_pos(bot, checkpoint.x, checkpoint.z);
    const targetPos = groundPos || checkpoint;

    console.log(
      `[${bot.username}] 📍 Checkpoint ${i + 1}/${checkpoints.length}: ` +
        `(${targetPos.x.toFixed(1)}, ${targetPos.y.toFixed(1)}, ${targetPos.z.toFixed(1)}) ` +
        `from (${bot.entity.position.x.toFixed(1)}, ${bot.entity.position.y.toFixed(1)}, ${bot.entity.position.z.toFixed(1)})`,
    );

    // Set the next goal immediately: setGoal() replaces any active goal, so
    // the bot rolls straight from one checkpoint into the next leg without a
    // stop-and-restart. Reach is judged horizontally: on uneven terrain the
    // bot's Y legitimately differs from the circle's Y, and a 3D check would
    // sit at the checkpoint until the timeout.
    const goal = new GoalNearXZ(
      targetPos.x,
      targetPos.z,
      CHECKPOINT_REACH_DISTANCE,
    );
    bot.pathfinder.setGoal(goal);

    // Wait until bot reaches checkpoint or timeout
    let checkCount = 0;
    let idleChecks = 0;
    const checkpointStartTime = Date.now();
    let progressPos = bot.entity.position.clone();
    let progressTime = checkpointStartTime;

    while (true) {
      const distance = bot.entity.position.xzDistanceTo(targetPos);
      const elapsed = Date.now() - checkpointStartTime;
      const moving = bot.pathfinder.isMoving();

      // Track XZ progress; if none over a window, we're stuck against terrain.
      const now = Date.now();
      if (
        bot.entity.position.xzDistanceTo(progressPos) >=
        CHECKPOINT_STUCK_MIN_PROGRESS
      ) {
        progressPos = bot.entity.position.clone();
        progressTime = now;
      } else if (
        elapsed > CHECKPOINT_IDLE_GRACE_MS &&
        now - progressTime > CHECKPOINT_STUCK_WINDOW_MS
      ) {
        console.log(
          `[${bot.username}] ↪️ No progress toward checkpoint ${i + 1} for ${((now - progressTime) / 1000).toFixed(1)}s ` +
            `(xz ${distance.toFixed(2)}, isMoving=${moving}); moving on`,
        );
        break;
      }

      // Log every 2 seconds (20 checks)
      if (checkCount % 20 === 0) {
        console.log(
          `[${bot.username}] 🔄 Moving to checkpoint ${i + 1}: xzDistance=${distance.toFixed(2)}, ` +
            `isMoving=${moving}, pos=(${bot.entity.position.x.toFixed(1)}, ${bot.entity.position.z.toFixed(1)}), ` +
            `elapsed=${(elapsed / 1000).toFixed(1)}s`,
        );
      }

      // The pathfinder judges GoalNearXZ on integer block coords, so it may
      // stop up to ~1 block short of the exact float target. Accept that
      // instead of standing still until the timeout.
      if (distance <= CHECKPOINT_REACH_DISTANCE + CHECKPOINT_REACH_SLACK) {
        console.log(
          `[${bot.username}] ✅ Reached checkpoint ${i + 1} in ${(elapsed / 1000).toFixed(1)}s (xz ${distance.toFixed(2)})`,
        );
        break;
      }

      // Pathfinder idle for a while after start = it thinks it's done or
      // found no path. Either way, waiting longer only adds a stationary
      // stretch; move on to the next checkpoint.
      idleChecks = !moving && elapsed > CHECKPOINT_IDLE_GRACE_MS ? idleChecks + 1 : 0;
      if (idleChecks >= CHECKPOINT_IDLE_CHECKS) {
        console.log(
          `[${bot.username}] ↪️ Pathfinder idle at checkpoint ${i + 1} after ${(elapsed / 1000).toFixed(1)}s ` +
            `(xz ${distance.toFixed(2)}); moving on`,
        );
        break;
      }

      if (elapsed > CHECKPOINT_TIMEOUT_MS) {
        console.log(
          `[${bot.username}] ⏰ Timeout at checkpoint ${i + 1} after ${(elapsed / 1000).toFixed(1)}s ` +
            `(xzDistance: ${distance.toFixed(2)} blocks, target: ${CHECKPOINT_REACH_DISTANCE} blocks)`,
        );
        break;
      }

      checkCount++;
      await sleep(100); // Check every 100ms
    }

    // Quick glance at the partner, hard-capped so the stop stays short. The
    // pathfinder takes the look direction back over as soon as the next goal
    // is set, which is what we want.
    console.log(`[${bot.username}] 👀 Glancing at ${otherBotName}`);
    await Promise.race([
      lookAtBot(bot, otherBotName, CAMERA_SPEED_DEGREES_PER_SEC, {
        randomized: false,
      }),
      sleep(CHECKPOINT_GLANCE_MAX_MS),
    ]);

    console.log(
      `[${bot.username}] ✅ Checkpoint ${i + 1} complete, continuing to next...`,
    );
  }

  // Clean up pathfinder after all checkpoints
  stopPathfinder(bot);

  console.log(
    `[${bot.username}] ✅ Orbit complete! Visited all ${checkpoints.length} checkpoints`,
  );
}

/**
 * Get orbit phase handler function
 */
function getOnOrbitPhaseFn(
  bot,
  rcon,
  sharedBotRng,
  coordinator,
  iterationID,
  otherBotName,
  episodeNum,
  episodeInstance,
  args,
  phaseDataOur,
) {
  return async (phaseDataOther) => {
    const startTime = Date.now();
    console.log(
      `[${bot.username}] 🌀 ORBIT EPISODE STARTING - Episode ${episodeNum}, Iteration ${iterationID}`,
    );

    coordinator.sendToOtherBot(
      `orbitPhase_${iterationID}`,
      phaseDataOur,
      episodeNum,
      `orbitPhase_${iterationID} beginning`,
    );

    // Step 1: Calculate shared midpoint between both bots
    const myPosition = phaseDataOur.position;
    const otherPosition = phaseDataOther.position;

    const sharedMidpoint = new Vec3(
      (myPosition.x + otherPosition.x) / 2,
      (myPosition.y + otherPosition.y) / 2,
      (myPosition.z + otherPosition.z) / 2,
    );

    console.log(
      `[${bot.username}] 📍 Shared midpoint: (${sharedMidpoint.x.toFixed(1)}, ` +
        `${sharedMidpoint.y.toFixed(1)}, ${sharedMidpoint.z.toFixed(1)})`,
    );

    // Step 2: Calculate orbit radius (half the distance between bots)
    const distanceBetweenBots = myPosition.distanceTo(otherPosition);
    const orbitRadius = distanceBetweenBots / 2;

    console.log(
      `[${bot.username}] 📏 Distance between bots: ${distanceBetweenBots.toFixed(2)} blocks`,
    );
    console.log(
      `[${bot.username}] ⭕ Orbit radius: ${orbitRadius.toFixed(2)} blocks`,
    );

    // Step 3: Calculate starting angle based on bot's current position
    const dx = myPosition.x - sharedMidpoint.x;
    const dz = myPosition.z - sharedMidpoint.z;
    const startAngle = Math.atan2(dz, dx);

    console.log(
      `[${bot.username}] 🎯 Starting angle: ${((startAngle * 180) / Math.PI).toFixed(1)}°`,
    );

    // Step 4: Generate checkpoint coordinates around the circle
    const checkpoints = calculateOrbitCheckpoints(
      sharedMidpoint,
      orbitRadius,
      NUM_CHECKPOINTS,
      startAngle,
    );

    console.log(
      `[${bot.username}] 📋 Generated ${checkpoints.length} checkpoints for orbit path`,
    );

    // Step 5 & 6: Execute orbit by traveling to checkpoints
    await executeOrbitWithCheckpoints(bot, otherBotName, checkpoints, rcon);

    // Transition to stop phase
    console.log(`[${bot.username}] Transitioning to stop phase...`);
    coordinator.onceEvent(
      "stopPhase",
      episodeNum,
      episodeInstance.getOnStopPhaseFn(
        bot,
        rcon,
        sharedBotRng,
        coordinator,
        otherBotName,
        episodeNum,
        args,
      ),
    );
    coordinator.sendToOtherBot(
      "stopPhase",
      phaseDataOur,
      episodeNum,
      `orbitPhase_${iterationID} end`,
    );

    const duration = (Date.now() - startTime) / 1000;
    console.log(
      `[${bot.username}] Orbit phase complete in ${duration.toFixed(1)}s`,
    );
  };
}

/**
 * Episode where both bots orbit around their shared midpoint, visiting checkpoints and making
 * eye contact at each. Orbit radius is half the distance between bots.
 * @extends BaseEpisode
 */
class OrbitEpisode extends BaseEpisode {
  static WORKS_IN_NON_FLAT_WORLD = true;

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
    console.log(`[${bot.username}] 🌀 Setting up orbit episode`);
    return {
      botPositionNew: botPosition,
      otherBotPositionNew: otherBotPosition,
    };
  }

  async entryPoint(
    bot,
    rcon,
    sharedBotRng,
    coordinator,
    iterationID,
    episodeNum,
    args,
  ) {
    const phaseDataOur = {
      position: bot.entity.position.clone(),
    };

    coordinator.onceEvent(
      `orbitPhase_${iterationID}`,
      episodeNum,
      getOnOrbitPhaseFn(
        bot,
        rcon,
        sharedBotRng,
        coordinator,
        iterationID,
        args.other_bot_name,
        episodeNum,
        this,
        args,
        phaseDataOur,
      ),
    );
    coordinator.sendToOtherBot(
      `orbitPhase_${iterationID}`,
      phaseDataOur,
      episodeNum,
      "teleportPhase end",
    );
  }

  async tearDownEpisode(
    bot,
    rcon,
    sharedBotRng,
    coordinator,
    episodeNum,
    args,
  ) {
    console.log(`[${bot.username}] 🧹 Cleaning up orbit episode`);
    stopPathfinder(bot);
    bot.setControlState("forward", false);
    bot.setControlState("back", false);
    bot.setControlState("left", false);
    bot.setControlState("right", false);
    bot.setControlState("sprint", false);
  }
}

module.exports = { OrbitEpisode };

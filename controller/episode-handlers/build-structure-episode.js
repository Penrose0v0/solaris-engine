// build-structure-episode.js - Collaborative building episode
const { Vec3 } = require("vec3");

const { buildPhase, rotateLocalToWorld, splitWorkByXAxis } = require("../primitives/building");
const { ensureBotHasEnough, unequipHand } = require("../primitives/items");
const {
  gotoWithTimeout,
  initializePathfinder,
  land_pos,
  sleep,
  stopPathfinder,
} = require("../primitives/movement");
const { GoalNearXZ } = require("../utils/bot-factory");
const { BaseEpisode } = require("./base-episode");

// Constants for building behavior
const ALL_STRUCTURE_TYPES = ["wall", "tower", "platform"];
// Structure selection weights (must sum to 1). Skewed toward platform: it
// yields the most camera translation (bigger footprint, collaborative build),
// while sparse structures (wall) tend to fail downstream motion checks.
const STRUCTURE_TYPE_WEIGHTS = { platform: 0.4, tower: 0.2, wall: 0.4 };
const INITIAL_EYE_CONTACT_MS = 1500; // Initial look duration
const BUILD_BLOCK_TYPES = ["stone", "cobblestone", "oak_planks", "bricks"];
const BLOCK_PLACE_DELAY_MS = 300; // Delay between placing blocks
const ORIENTATION = 0; // Only 0° supported for now

// Camera-motion interleaving: after every few placed blocks the bot walks an
// arc around the structure-in-progress before continuing. The build itself
// (what/where/how many blocks) is unchanged; this only adds translation
// around it so the recording isn't a stationary camera.
const ARC_WALK_BLOCKS_PER_LEG = [2]; // blocks placed between arc walks
const ARC_WALK_SMALL_STRUCTURE_BLOCKS = 3; // structures this small walk after every block
// Each leg is several 90° hops so the pathfinder actually circles the
// structure instead of cutting straight through/past it. 3 hops = 270°: a
// real lap's worth of translation, ending on a different side.
const ARC_WALK_HOP_RAD = Math.PI / 2;
const ARC_WALK_HOPS_PER_LEG = 3;
const ARC_WALK_MIN_RADIUS = 4.0; // blocks from structure centre
const ARC_WALK_RADIUS_MARGIN = 2.5; // beyond the structure's half-extent
const ARC_WALK_HOP_TIMEOUT_MS = 4000; // give up on a hop rather than stall
// After finishing its own blocks a bot waits for the partner (the stop phase
// is triggered by whichever bot finishes last). Keep circling meanwhile
// instead of standing still; hard cap so it can never outlive the episode.
const WAIT_FOR_PARTNER_MAX_MS = 120000;
// Small structures (tower) finish in ~15s, which is too few frames for
// downstream (needs >= ~800 frames @20fps). After building, keep circling
// until the build phase has run at least this long before handing off to the
// stop phase. Recording only; the build itself is unchanged.
const SMALL_STRUCTURE_MIN_PHASE_MS = 42000;

/**
 * Generate blueprint for a wall structure (5 blocks wide × 2 blocks tall)
 * @param {Object} options - Configuration options
 * @param {string} options.blockType - Block type to use
 * @returns {Array<Object>} Array of {x, y, z, block, phase, placementOrder}
 */
function makeWallBlueprint(options = {}) {
  const { blockType = "stone" } = options;
  const blueprint = [];

  const width = 5;
  const height = 2;

  // Build bottom-up, left-to-right
  let order = 0;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      blueprint.push({
        x,
        y,
        z: 0,
        block: blockType,
        phase: "wall",
        placementOrder: order++,
        data: null,
      });
    }
  }

  return blueprint;
}

/**
 * Generate blueprint for a tower structure (single column, 2 blocks tall)
 * @param {Object} options - Configuration options
 * @param {string} options.blockType - Block type to use
 * @returns {Array<Object>} Array of {x, y, z, block, phase, placementOrder}
 */
function makeTowerBlueprint(options = {}) {
  const { blockType = "stone" } = options;
  const blueprint = [];

  const height = 2;

  // Build bottom-up
  for (let y = 0; y < height; y++) {
    blueprint.push({
      x: 0,
      y,
      z: 0,
      block: blockType,
      phase: "tower",
      placementOrder: y,
      data: null,
    });
  }

  return blueprint;
}

/**
 * Generate blueprint for a platform structure (4×4 grid)
 * @param {Object} options - Configuration options
 * @param {string} options.blockType - Block type to use
 * @returns {Array<Object>} Array of {x, y, z, block, phase, placementOrder}
 */
function makePlatformBlueprint(options = {}) {
  const { blockType = "stone" } = options;
  const blueprint = [];

  const width = 4;
  const depth = 4;

  // Build edge-to-center spiral (same as house floor)
  let order = 0;
  let minX = 0,
    maxX = width - 1;
  let minZ = 0,
    maxZ = depth - 1;

  while (minX <= maxX && minZ <= maxZ) {
    // Top edge (left to right)
    for (let x = minX; x <= maxX; x++) {
      blueprint.push({
        x,
        y: 0,
        z: minZ,
        block: blockType,
        phase: "platform",
        placementOrder: order++,
        data: null,
      });
    }
    minZ++;

    // Right edge (top to bottom)
    for (let z = minZ; z <= maxZ; z++) {
      blueprint.push({
        x: maxX,
        y: 0,
        z,
        block: blockType,
        phase: "platform",
        placementOrder: order++,
        data: null,
      });
    }
    maxX--;

    // Bottom edge (right to left)
    if (minZ <= maxZ) {
      for (let x = maxX; x >= minX; x--) {
        blueprint.push({
          x,
          y: 0,
          z: maxZ,
          block: blockType,
          phase: "platform",
          placementOrder: order++,
          data: null,
        });
      }
      maxZ--;
    }

    // Left edge (bottom to top)
    if (minX <= maxX) {
      for (let z = maxZ; z >= minZ; z--) {
        blueprint.push({
          x: minX,
          y: 0,
          z,
          block: blockType,
          phase: "platform",
          placementOrder: order++,
          data: null,
        });
      }
      minX++;
    }
  }

  return blueprint;
}

/**
 * Pick a structure type using STRUCTURE_TYPE_WEIGHTS with a single shared-RNG
 * draw (so both bots agree). Types are iterated in sorted order for
 * determinism; falls back to the last type on floating-point rounding.
 * @param {Function} sharedBotRng - Shared random number generator
 * @returns {string} One of ALL_STRUCTURE_TYPES
 */
function pickWeightedStructureType(sharedBotRng) {
  const types = ALL_STRUCTURE_TYPES.slice().sort();
  const r = sharedBotRng();
  let acc = 0;
  for (const t of types) {
    acc += STRUCTURE_TYPE_WEIGHTS[t] ?? 0;
    if (r < acc) return t;
  }
  return types[types.length - 1];
}

/**
 * Walk an arc around the structure so the camera translates between build
 * chunks. Target = point on a circle around the structure's XZ centre, at the
 * bot's current angle + ARC_WALK_ANGLE_RAD (so successive legs keep circling
 * in the same direction). Best-effort: a timeout or pathing failure just
 * returns and building resumes.
 * @param {*} bot - Mineflayer bot instance
 * @param {Vec3} center - Structure XZ centre (y is a fallback ground level)
 * @param {number} radius - Arc radius in blocks
 * @param {number} legIndex - Which leg this is (for logging)
 */
async function walkArcAroundStructure(
  bot,
  center,
  radius,
  legIndex,
  shouldStop = () => false,
) {
  const start = Date.now();
  let angle = Math.atan2(
    bot.entity.position.z - center.z,
    bot.entity.position.x - center.x,
  );
  console.log(
    `[${bot.username}] 🚶 Arc leg ${legIndex}: ${ARC_WALK_HOPS_PER_LEG} x ${((ARC_WALK_HOP_RAD * 180) / Math.PI).toFixed(0)}° around structure [r=${radius.toFixed(1)}]`,
  );

  for (let h = 0; h < ARC_WALK_HOPS_PER_LEG; h++) {
    if (shouldStop()) break;
    angle += ARC_WALK_HOP_RAD;
    const tx = center.x + radius * Math.cos(angle);
    const tz = center.z + radius * Math.sin(angle);
    const ground = land_pos(bot, tx, tz);
    const ty = ground ? ground.y : center.y;

    // Race the hop against the stop signal so a stop phase (or partner
    // finishing) interrupts within ~100ms rather than after the hop.
    let stopped = false;
    const stopWatch = (async () => {
      while (!stopped && !shouldStop()) await sleep(100);
      if (!stopped && bot.pathfinder) bot.pathfinder.setGoal(null);
    })();
    try {
      await gotoWithTimeout(bot, new GoalNearXZ(tx, tz, 1.0), {
        timeoutMs: ARC_WALK_HOP_TIMEOUT_MS,
      });
    } catch (err) {
      console.log(
        `[${bot.username}] ⚠️ Arc leg ${legIndex} hop ${h + 1} to (${tx.toFixed(1)}, ${ty.toFixed(1)}, ${tz.toFixed(1)}) cut short: ${err.message}`,
      );
    } finally {
      stopped = true;
      await stopWatch;
    }
  }
  console.log(
    `[${bot.username}] ✅ Arc leg ${legIndex} done in ${((Date.now() - start) / 1000).toFixed(1)}s`,
  );
}

/**
 * Split an ordered target list into consecutive chunks whose sizes alternate
 * through ARC_WALK_BLOCKS_PER_LEG (e.g. 3,2,3,2,...).
 * @param {Array<Object>} targets - Targets already sorted in placement order
 * @returns {Array<Array<Object>>}
 */
function chunkTargetsForArcWalks(targets) {
  const chunks = [];
  const small = targets.length <= ARC_WALK_SMALL_STRUCTURE_BLOCKS;
  let i = 0;
  let k = 0;
  while (i < targets.length) {
    const size = small
      ? 1
      : ARC_WALK_BLOCKS_PER_LEG[k % ARC_WALK_BLOCKS_PER_LEG.length];
    chunks.push(targets.slice(i, i + size));
    i += size;
    k++;
  }
  return chunks;
}

/**
 * Get the phase function for building episodes
 * @param {*} bot - Mineflayer bot instance
 * @param {Object} rcon - RCON connection
 * @param {Function} sharedBotRng - Shared random number generator
 * @param {*} coordinator - Bot coordinator instance
 * @param {number} iterationID - Iteration ID
 * @param {number} episodeNum - Episode number
 * @param {Object} episodeInstance - Episode instance
 * @param {Object} args - Configuration arguments
 * @param {string} structureType - Type of structure ('wall', 'tower', 'platform')
 * @param {Object} phaseDataOur - Phase data for this bot (contains position)
 * @returns {Function} Phase function
 */
function getOnBuildPhaseFn(
  bot,
  rcon,
  sharedBotRng,
  coordinator,
  iterationID,
  episodeNum,
  episodeInstance,
  args,
  structureType = "wall",
  phaseDataOur,
) {
  return async function onBuildPhase(phaseDataOther) {
    const phaseStartTime = Date.now();
    coordinator.sendToOtherBot(
      `buildPhase_${iterationID}`,
      phaseDataOur,
      episodeNum,
      `buildPhase_${iterationID} beginning`,
    );

    console.log(`[${bot.username}] 🚀 Starting BUILD phase ${iterationID}`);

    // STEP 1: Bots spawn (already done by teleport phase)
    console.log(`[${bot.username}] ✅ STEP 1: Bot spawned`);

    // STEP 2: Initial eye contact
    console.log(
      `[${bot.username}] 👀 STEP 2: Making eye contact with ${args.other_bot_name}...`,
    );
    try {
      const otherEntity = bot.players[args.other_bot_name]?.entity;
      if (otherEntity) {
        const targetPos = otherEntity.position.offset(0, otherEntity.height, 0);
        await bot.lookAt(targetPos, false);
        await sleep(INITIAL_EYE_CONTACT_MS);
      }
    } catch (lookError) {
      console.log(
        `[${bot.username}] ⚠️ Could not look at other bot: ${lookError.message}`,
      );
    }

    // STEP 3: Determine world origin for structure (midpoint between bots)
    console.log(
      `[${bot.username}] 📐 STEP 3: Planning structure ${structureType}...`,
    );

    // Reconstruct Vec3 from received position data
    const botPos = new Vec3(
      phaseDataOur.position.x,
      phaseDataOur.position.y,
      phaseDataOur.position.z,
    ).floored();
    const otherBotPos = new Vec3(
      phaseDataOther.position.x,
      phaseDataOther.position.y,
      phaseDataOther.position.z,
    ).floored();

    // For wall and tower: each bot builds at their own position
    // For platform: build at midpoint (collaborative)
    let worldOrigin;
    if (structureType === "wall" || structureType === "tower") {
      // Each bot builds their own structure at their spawn location
      worldOrigin = botPos.clone();
      console.log(
        `[${bot.username}] 🏗️ Building individual ${structureType} at bot position: (${worldOrigin.x}, ${worldOrigin.y}, ${worldOrigin.z})`,
      );
    } else {
      // Platform: build at midpoint between bots (collaborative)
      // Use the higher Y level between bots to ensure consistent elevation
      const maxBotY = Math.max(botPos.y, otherBotPos.y);
      worldOrigin = new Vec3(
        Math.floor((botPos.x + otherBotPos.x) / 2),
        Math.floor(maxBotY), // Use higher Y level for consistency (lower bot will pathfind up)
        Math.floor((botPos.z + otherBotPos.z) / 2),
      );
      console.log(
        `[${bot.username}] 🏗️ Platform origin (midpoint at higher Y): (${worldOrigin.x}, ${worldOrigin.y}, ${worldOrigin.z}) [maxY: ${maxBotY}]`,
      );
    }

    // STEP 4: Generate blueprint and convert to world coordinates
    console.log(`[${bot.username}] 📋 STEP 4: Generating blueprint...`);

    // Pick block type using shared RNG
    const blockType =
      BUILD_BLOCK_TYPES[Math.floor(sharedBotRng() * BUILD_BLOCK_TYPES.length)];
    console.log(`[${bot.username}] 📦 Block type: ${blockType}`);

    let blueprint;
    if (structureType === "wall") {
      blueprint = makeWallBlueprint({ blockType });
    } else if (structureType === "tower") {
      blueprint = makeTowerBlueprint({ blockType });
    } else if (structureType === "platform") {
      blueprint = makePlatformBlueprint({ blockType });
    }

    // Convert all local coords to world coords
    const worldTargets = blueprint.map((target) => ({
      ...target,
      worldPos: rotateLocalToWorld(target, worldOrigin, ORIENTATION),
    }));

    console.log(`[${bot.username}]    Total blocks: ${worldTargets.length}`);

    // STEP 5: Assign work (split only for platform, full structure for wall/tower)
    console.log(`[${bot.username}] 🔀 STEP 5: Assigning work...`);

    let myTargets;

    if (structureType === "platform") {
      // Platform: Split work between bots
      const { alphaTargets, bravoTargets } = splitWorkByXAxis(
        worldTargets,
        args.bot_name,
        args.other_bot_name,
      );

      // Determine which half based on proximity to structure origin
      const botIsOnWestSide = botPos.x < worldOrigin.x;
      const otherBotIsOnWestSide = otherBotPos.x < worldOrigin.x;

      // If conflict (both bots on same side), use bot identity as tie-breaker
      if (botIsOnWestSide === otherBotIsOnWestSide) {
        // Tie! Use alphabetical order: Alpha gets west, Bravo gets east
        const isAlphaBot = bot.username < args.other_bot_name;
        myTargets = isAlphaBot ? alphaTargets : bravoTargets;
        console.log(
          `[${bot.username}] ⚠️ Both bots on ${botIsOnWestSide ? "WEST" : "EAST"} side - using tie-breaker (${isAlphaBot ? "WEST" : "EAST"} half)`,
        );
      } else {
        // No conflict - use proximity-based assignment
        myTargets = botIsOnWestSide ? alphaTargets : bravoTargets;
        console.log(
          `[${bot.username}] ✅ Using proximity-based assignment (${botIsOnWestSide ? "WEST" : "EAST"} half)`,
        );
      }

      console.log(
        `[${bot.username}] 📍 Spawn position: (${botPos.x}, ${botPos.z}), Platform origin: (${worldOrigin.x}, ${worldOrigin.z})`,
      );
      console.log(
        `[${bot.username}] 🏗️ Assigned ${myTargets.length}/${worldTargets.length} blocks (collaborative build)`,
      );
    } else {
      // Wall/Tower: Each bot builds their own complete structure
      myTargets = worldTargets;
      console.log(
        `[${bot.username}] ✅ Building complete ${structureType} independently`,
      );
      console.log(
        `[${bot.username}] 📍 Structure position: (${worldOrigin.x}, ${worldOrigin.y}, ${worldOrigin.z})`,
      );
      console.log(
        `[${bot.username}] 🏗️ Assigned ${myTargets.length} blocks (individual build)`,
      );
    }

    // STEP 6: Initialize pathfinder for building
    console.log(`[${bot.username}] 🚶 STEP 6: Initializing pathfinder...`);
    initializePathfinder(bot, {
      allowSprinting: false,
      allowParkour: true,
      canDig: false,
      allowEntityDetection: true,
    });

    // STEP 7: Build the structure, walking an arc around it between chunks
    // of 2-3 blocks so the camera keeps translating (see ARC_WALK_* above).
    console.log(`[${bot.username}] 🏗️ STEP 7: Building structure...`);

    // Structure XZ centre + radius for the arc walks, from the full blueprint
    // so both bots circle the same thing on collaborative builds.
    const xs = worldTargets.map((t) => t.worldPos.x);
    const zs = worldTargets.map((t) => t.worldPos.z);
    const structureCenter = new Vec3(
      (Math.min(...xs) + Math.max(...xs)) / 2 + 0.5,
      worldOrigin.y,
      (Math.min(...zs) + Math.max(...zs)) / 2 + 0.5,
    );
    const halfExtent =
      Math.max(
        Math.max(...xs) - Math.min(...xs),
        Math.max(...zs) - Math.min(...zs),
      ) /
        2 +
      0.5;
    const arcRadius = Math.max(
      ARC_WALK_MIN_RADIUS,
      halfExtent + ARC_WALK_RADIUS_MARGIN,
    );

    // buildPhase sorts by placementOrder internally; sort here too so the
    // chunks follow the same order it would have used.
    const orderedTargets = myTargets
      .slice()
      .sort((a, b) => (a.placementOrder ?? 0) - (b.placementOrder ?? 0));
    const chunks = chunkTargetsForArcWalks(orderedTargets);
    console.log(
      `[${bot.username}] 🔁 ${orderedTargets.length} blocks in ${chunks.length} chunk(s), arc radius ${arcRadius.toFixed(1)}`,
    );

    let buildResult = { success: 0, failed: 0, aborted: false };
    try {
      for (let c = 0; c < chunks.length; c++) {
        if (bot._episodeStopping) {
          buildResult.aborted = true;
          break;
        }

        // Walk an arc before every chunk except the first (the bot is
        // already at the build site when the phase starts) — except for
        // small structures (tower), whose episodes are so short that the
        // fixed setup head would otherwise dominate; give those a lap first.
        const smallStructure =
          orderedTargets.length <= ARC_WALK_SMALL_STRUCTURE_BLOCKS;
        if (c > 0 || smallStructure) {
          await walkArcAroundStructure(
            bot,
            structureCenter,
            arcRadius,
            c,
            () => bot._episodeStopping,
          );
          if (bot._episodeStopping) {
            buildResult.aborted = true;
            break;
          }
        }

        const chunkResult = await buildPhase(bot, chunks[c], {
          args: args,
          delayMs: BLOCK_PLACE_DELAY_MS,
          shouldAbort: () => bot._episodeStopping,
        });
        buildResult.success += chunkResult.success;
        buildResult.failed += chunkResult.failed;
        if (chunkResult.aborted) {
          buildResult.aborted = true;
          break;
        }
      }

      if (buildResult.aborted) {
        console.log(`[${bot.username}] 🛑 Build aborted due to stop request`);
        stopPathfinder(bot);
        return;
      }

      // Check for catastrophic failure (more than 50% failed)
      if (buildResult.failed > myTargets.length * 0.5) {
        console.log(
          `[${bot.username}] ❌ Build failed significantly: ${buildResult.failed}/${myTargets.length} blocks failed`,
        );
        throw new Error(
          `Build failed: ${buildResult.failed}/${myTargets.length} blocks failed`,
        );
      }

      console.log(`[${bot.username}] ✅ Build complete!`);
    } catch (buildError) {
      console.error(
        `[${bot.username}] ❌ Building failed: ${buildError.message}`,
      );

      // Stop pathfinder immediately
      if (bot.pathfinder) {
        bot.pathfinder.setGoal(null);
      }
      stopPathfinder(bot);

      // Re-throw the error so episode system handles it properly
      throw buildError;
    }

    // STEP 8: Stop pathfinder
    stopPathfinder(bot);

    if (bot._episodeStopping) {
      console.log(
        `[${bot.username}] 🛑 Stop phase already in progress, skipping final eye contact`,
      );
      return;
    }

    // STEP 9: Final eye contact
    console.log(`[${bot.username}] 👀 STEP 9: Final eye contact...`);
    try {
      const otherEntity = bot.players[args.other_bot_name]?.entity;
      if (otherEntity) {
        const targetPos = otherEntity.position.offset(0, otherEntity.height, 0);
        await bot.lookAt(targetPos, false);
        await sleep(INITIAL_EYE_CONTACT_MS);
      }
    } catch (lookError) {
      console.log(
        `[${bot.username}] ⚠️ Could not look at other bot: ${lookError.message}`,
      );
    }

    console.log(`[${bot.username}] ✅ BUILD phase complete!`);

    // Small structures: pad the recording by circling until the phase has
    // lasted SMALL_STRUCTURE_MIN_PHASE_MS, *before* signalling stop.
    if (orderedTargets.length <= ARC_WALK_SMALL_STRUCTURE_BLOCKS) {
      const minPhaseReached = () =>
        Date.now() - phaseStartTime >= SMALL_STRUCTURE_MIN_PHASE_MS;
      let padLeg = 0;
      while (!minPhaseReached() && !bot._episodeStopping) {
        padLeg++;
        console.log(
          `[${bot.username}] ⏱️ Small structure: padding recording, circling (leg ${padLeg}, ` +
            `${((Date.now() - phaseStartTime) / 1000).toFixed(0)}s / ${SMALL_STRUCTURE_MIN_PHASE_MS / 1000}s)`,
        );
        await walkArcAroundStructure(
          bot,
          structureCenter,
          arcRadius,
          chunks.length + padLeg,
          () => bot._episodeStopping || minPhaseReached(),
        );
      }
      stopPathfinder(bot);
    }

    // Transition to stop phase
    // The stop phase fires when the *last* bot finishes (the coordinator drops
    // events that arrive before a listener exists, so the first finisher's
    // stopPhase is only answered once the partner is done and re-broadcasts).
    // Wrap the handler to learn when that happens, and keep circling the
    // structure meanwhile instead of standing still.
    let partnerDone = false;
    const onStopPhase = episodeInstance.getOnStopPhaseFn(
      bot,
      rcon,
      sharedBotRng,
      coordinator,
      args.other_bot_name,
      episodeNum,
      args,
    );
    coordinator.onceEvent("stopPhase", episodeNum, async (...eventArgs) => {
      partnerDone = true;
      return onStopPhase(...eventArgs);
    });
    coordinator.sendToOtherBot(
      "stopPhase",
      phaseDataOur,
      episodeNum,
      `buildPhase_${iterationID} end`,
    );

    const waitStart = Date.now();
    const shouldStopWaiting = () =>
      partnerDone ||
      bot._episodeStopping ||
      Date.now() - waitStart > WAIT_FOR_PARTNER_MAX_MS;
    let waitLeg = 0;
    while (!shouldStopWaiting()) {
      waitLeg++;
      console.log(
        `[${bot.username}] ⏳ Waiting for partner; circling structure (leg ${waitLeg})`,
      );
      await walkArcAroundStructure(
        bot,
        structureCenter,
        arcRadius,
        chunks.length + waitLeg,
        shouldStopWaiting,
      );
    }
    stopPathfinder(bot);
    if (waitLeg > 0) {
      console.log(
        `[${bot.username}] ✅ Partner finished after ${((Date.now() - waitStart) / 1000).toFixed(1)}s of circling (${waitLeg} leg(s))`,
      );
    }

    return buildResult;
  };
}

/**
 * Episode for collaborative structure building (wall, tower, or platform). Structure type is
 * chosen per episode via shared RNG; wall/tower are built individually, platform is split by X-axis.
 * @extends BaseEpisode
 */
class BuildStructureEpisode extends BaseEpisode {
  static INIT_MIN_BOTS_DISTANCE = 8;
  static INIT_MAX_BOTS_DISTANCE = 15;
  static WORKS_IN_NON_FLAT_WORLD = true;

  constructor(sharedBotRng) {
    super();
    this.structureType = pickWeightedStructureType(sharedBotRng);
    // Surfaces as a top-level `structure_type` field in *_episode_info.json
    // so downstream can split buildStructure episodes by structure.
    this._episodeMetadata = { structure_type: this.structureType };
  }

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
    console.log(
      `[${bot.username}] 🏗️ Setting up structure building episode...`,
    );

    // Give all block types (we'll pick one during build phase using shared RNG)
    console.log(`[${bot.username}] 📦 Giving building materials...`);
    for (const blockType of BUILD_BLOCK_TYPES) {
      await ensureBotHasEnough(bot, rcon, blockType, 64);
    }

    await unequipHand(bot);

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
      `buildPhase_${iterationID}`,
      episodeNum,
      getOnBuildPhaseFn(
        bot,
        rcon,
        sharedBotRng,
        coordinator,
        iterationID,
        episodeNum,
        this,
        args,
        this.structureType,
        phaseDataOur,
      ),
    );
    coordinator.sendToOtherBot(
      `buildPhase_${iterationID}`,
      phaseDataOur,
      episodeNum,
      "entryPoint end",
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
    console.log(
      `[${bot.username}] 🧹 Cleaning up structure building episode...`,
    );
    // Clean up pathfinder if still active
    if (bot.pathfinder) {
      stopPathfinder(bot);
    }
  }
}

module.exports = { BuildStructureEpisode };

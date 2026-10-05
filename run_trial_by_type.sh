#!/bin/bash
# Trial GT data production, grouped by episode type.
#
# For each episode type, runs a small collection (normal world, overhead
# cameras on), aligns the recordings, and flattens into
#   <OUT_ROOT>/<type>/dataset/{videos,videos_overhead,trajectories,episode_info,first_frames}
# so downstream can score which types are usable (enough camera translation,
# recognizable texture). Per-frame pose (x/y/z/yaw/pitch) lives in
# trajectories/*.json.
#
# A type that fails is logged to <OUT_ROOT>/trial_summary.txt and skipped;
# no episode behavior/parameters are changed here.

# NOTE: no `set -e` — one type failing must not abort the others.

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
TYPES="chase,orbit,straightLineWalk,translationEval,buildStructure,placeAndMine"
INSTANCES=2          # parallel worlds per type
NUM_EPISODES=5       # episodes per instance -> INSTANCES*NUM_EPISODES per type (~10)
FPS=20
GPU_COUNT=1
OUT_ROOT="output/trial"
CACHE_BASE="camera_cache"

while [[ $# -gt 0 ]]; do
  case $1 in
    --types)        TYPES="$2";        shift 2 ;;
    --instances)    INSTANCES="$2";    shift 2 ;;
    --num-episodes) NUM_EPISODES="$2"; shift 2 ;;
    --fps)          FPS="$2";          shift 2 ;;
    --gpu-count)    GPU_COUNT="$2";    shift 2 ;;
    --output-dir)   OUT_ROOT="$2";     shift 2 ;;
    -h|--help)
      echo "Usage: $0 [OPTIONS]"
      echo "  --types a,b,c     Comma-separated episode types (default: $TYPES)"
      echo "  --instances N     Parallel worlds per type (default: $INSTANCES)"
      echo "  --num-episodes N  Episodes per instance (default: $NUM_EPISODES)"
      echo "  --fps N           Camera fps + action rate (default: $FPS)"
      echo "  --gpu-count N     GPUs to spread instances across (default: $GPU_COUNT)"
      echo "  --output-dir DIR  Root output dir; one subdir per type (default: $OUT_ROOT)"
      exit 0
      ;;
    *) echo "Unknown option: $1 (use --help)" >&2; exit 1 ;;
  esac
done

source /storage/home/yang.hu/miniforge3/etc/profile.d/conda.sh
conda activate SolarisEngine
cd "$(dirname "$0")"

mkdir -p "$OUT_ROOT"
SUMMARY="$OUT_ROOT/trial_summary.txt"
echo "trial started $(date -Is)  types=$TYPES instances=$INSTANCES episodes/instance=$NUM_EPISODES" > "$SUMMARY"

IFS=',' read -ra TYPE_LIST <<< "$TYPES"
for TYPE in "${TYPE_LIST[@]}"; do
  TYPE="${TYPE// /}"
  BATCH_DIR="$OUT_ROOT/$TYPE"
  COMPOSE_DIR="$BATCH_DIR/compose_configs"
  echo
  echo "############################################################"
  echo "### episode type: $TYPE  ->  $BATCH_DIR"
  echo "############################################################"

  # Any leftover containers from a previous type would collide on ports.
  docker ps -q --filter "name=docker-compose-" | xargs -r docker rm -f >/dev/null 2>&1

  python3 generate_compose.py \
      --compose_dir "$COMPOSE_DIR" \
      --base_port 25590 --base_rcon_port 25700 \
      --act_recorder_port 8110 --coord_port 8120 \
      --data_dir "$BATCH_DIR/data" \
      --output_dir "$BATCH_DIR/output" \
      --camera_output_alpha_base "$BATCH_DIR/camera/output_alpha" \
      --camera_output_bravo_base "$BATCH_DIR/camera/output_bravo" \
      --camera_data_alpha_base "$CACHE_BASE/data_alpha" \
      --camera_data_bravo_base "$CACHE_BASE/data_bravo" \
      --enable_overhead_cameras 1 \
      --camera_output_overhead_alpha_base "$BATCH_DIR/camera/output_overhead_alpha" \
      --camera_output_overhead_bravo_base "$BATCH_DIR/camera/output_overhead_bravo" \
      --camera_data_overhead_alpha_base "$CACHE_BASE/data_overhead_alpha" \
      --camera_data_overhead_bravo_base "$CACHE_BASE/data_overhead_bravo" \
      --num_normal_world "$INSTANCES" \
      --episode_types "$TYPE" \
      --smoke_test 1 \
      --num_episodes "$NUM_EPISODES" \
      --fps "$FPS" \
      --gpu_count "$GPU_COUNT" \
      --viewer_rendering_disabled 1 \
      --gpu_mode egl
  if [[ $? -ne 0 ]]; then
    echo "$TYPE: FAILED (generate_compose)" | tee -a "$SUMMARY"
    continue
  fi

  python3 orchestrate.py start --compose-dir "$COMPOSE_DIR" --logs-dir "$BATCH_DIR/logs"
  # orchestrate exits 0 even if controllers fail; judge by produced episodes below.
  docker ps -q --filter "name=docker-compose-" | xargs -r docker rm -f >/dev/null 2>&1

  docker run --rm -v "$(pwd)/$BATCH_DIR:/workspace" alpine chown -R "$(id -u):$(id -g)" /workspace
  docker run --rm -v "$(pwd)/$CACHE_BASE:/workspace" alpine chown -R "$(id -u):$(id -g)" /workspace

  n_raw=$(ls "$BATCH_DIR"/output/*_episode_info.json 2>/dev/null | wc -l)
  if [[ "$n_raw" -eq 0 ]]; then
    echo "$TYPE: FAILED (0 episodes recorded; see $BATCH_DIR/logs)" | tee -a "$SUMMARY"
    continue
  fi

  python3 orchestrate.py postprocess --compose-dir "$COMPOSE_DIR" --workers 16 --output-dir "$BATCH_DIR/aligned"
  python3 postprocess/flatten_overhead_dataset.py \
      --source-dir "$BATCH_DIR" --destination-dir "$BATCH_DIR/dataset" --batch-name "$TYPE"

  n_ok=$(ls "$BATCH_DIR"/dataset/trajectories/*.json 2>/dev/null | wc -l)
  echo "$TYPE: $((n_raw / 2)) episodes recorded, $((n_ok / 2)) passed alignment  ->  $BATCH_DIR/dataset" | tee -a "$SUMMARY"
done

echo
echo "=== Trial summary ($SUMMARY) ==="
cat "$SUMMARY"

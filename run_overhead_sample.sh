#!/bin/bash
set -e

# Full-chain data collection with overhead (top-down) cameras:
#   collect -> align/postprocess (H.264, VS Code-playable) -> flatten to a
#   solaris-new2-style dataset directory (plus *_overhead.mp4 per episode).

# ---------------------------------------------------------------------------
# Hyperparameters (defaults: ~1000-episode run across all 8 GPUs)
# ---------------------------------------------------------------------------
INSTANCES=16         # parallel worlds; NVENC fits 2 instances/GPU with
                     # overhead cameras on, 4/GPU with them off
NUM_EPISODES=63      # episodes per instance (16 * 63 = 1008 total)
FPS=20               # camera fps; action-recording interval is derived from
                     # it (1000/fps ms) so actions and frames stay 1:1
OVERHEAD=1           # 1 = also record a top-down camera above each player,
                     # 0 = first-person cameras only
GPU_COUNT=8          # GPUs to spread instances across (round-robin)
BATCH_DIR="output/overhead_1k"

# Persistent Minecraft client cache, shared across batches so the ~1GB
# client install is downloaded once per camera slot, not once per run.
CACHE_BASE="camera_cache"

# Parse CLI args (each flag overrides the default above)
while [[ $# -gt 0 ]]; do
  case $1 in
    --instances)    INSTANCES="$2";    shift 2 ;;
    --num-episodes) NUM_EPISODES="$2"; shift 2 ;;
    --fps)          FPS="$2";          shift 2 ;;
    --overhead)     OVERHEAD="$2";     shift 2 ;;
    --gpu-count)    GPU_COUNT="$2";    shift 2 ;;
    --output-dir)   BATCH_DIR="$2";    shift 2 ;;
    -h|--help)
      echo "Usage: $0 [OPTIONS]"
      echo "  --instances N     Parallel worlds (default: 16; NVENC max per GPU:"
      echo "                    2 with overhead cameras, 4 without)"
      echo "  --num-episodes N  Episodes per instance (default: 63)"
      echo "  --fps N           Camera fps + action rate (default: 20)"
      echo "  --overhead 0|1    Record top-down cameras too (default: 1)"
      echo "  --gpu-count N     GPUs to spread instances across (default: 8)"
      echo "  --output-dir DIR  Batch output directory (default: output/overhead_1k)"
      exit 0
      ;;
    *) echo "Unknown option: $1 (use --help)" >&2; exit 1 ;;
  esac
done

OVERHEAD_ARGS=()
CAMERA_KINDS=(data_alpha data_bravo)
if [[ "$OVERHEAD" == "1" ]]; then
  CAMERA_KINDS+=(data_overhead_alpha data_overhead_bravo)
  OVERHEAD_ARGS=(
    --enable_overhead_cameras 1
    --camera_output_overhead_alpha_base "$BATCH_DIR/camera/output_overhead_alpha"
    --camera_output_overhead_bravo_base "$BATCH_DIR/camera/output_overhead_bravo"
    --camera_data_overhead_alpha_base "$CACHE_BASE/data_overhead_alpha"
    --camera_data_overhead_bravo_base "$CACHE_BASE/data_overhead_bravo"
  )
fi

COMPOSE_DIR="$BATCH_DIR/compose_configs"
LOGS_DIR="$BATCH_DIR/logs"
DATASET_DIR="$BATCH_DIR/dataset"

source /storage/home/yang.hu/miniforge3/etc/profile.d/conda.sh
conda activate SolarisEngine

cd "$(dirname "$0")"

echo "Config: instances=$INSTANCES episodes/instance=$NUM_EPISODES fps=$FPS overhead=$OVERHEAD gpus=$GPU_COUNT out=$BATCH_DIR"

echo "=== [0/5] Warming Minecraft client caches ==="
# Seed empty per-instance cache dirs by hardlinking the immutable parts
# (assets/libraries/versions/...) from any already-populated slot, so a
# fresh 64-slot run doesn't trigger 64 simultaneous ~1GB downloads.
SEED=""
for jar in "$CACHE_BASE"/*/*/.minecraft/versions/1.21/1.21.jar; do
  [ -f "$jar" ] && SEED="${jar%/versions/1.21/1.21.jar}" && break
done
if [ -n "$SEED" ]; then
  seeded=0
  for kind in "${CAMERA_KINDS[@]}"; do
    for i in $(seq 0 $((INSTANCES - 1))); do
      dst="$CACHE_BASE/$kind/$i/.minecraft"
      if [ ! -f "$dst/versions/1.21/1.21.jar" ]; then
        mkdir -p "$dst"
        for d in assets libraries versions downloads runtime; do
          [ -e "$SEED/$d" ] && [ ! -e "$dst/$d" ] && cp -al "$SEED/$d" "$dst/$d"
        done
        seeded=$((seeded + 1))
      fi
    done
  done
  echo "Seeded $seeded cache slot(s) from $SEED"
else
  echo "No populated cache found under $CACHE_BASE; clients will download on first launch"
fi

echo "=== [1/5] Generating compose configs ==="
python3 generate_compose.py \
    --compose_dir "$COMPOSE_DIR" \
    --base_port 25590 \
    --base_rcon_port 25700 \
    --act_recorder_port 8110 \
    --coord_port 8120 \
    --data_dir "$BATCH_DIR/data" \
    --output_dir "$BATCH_DIR/output" \
    --camera_output_alpha_base "$BATCH_DIR/camera/output_alpha" \
    --camera_output_bravo_base "$BATCH_DIR/camera/output_bravo" \
    --camera_data_alpha_base "$CACHE_BASE/data_alpha" \
    --camera_data_bravo_base "$CACHE_BASE/data_bravo" \
    "${OVERHEAD_ARGS[@]}" \
    --instances "$INSTANCES" \
    --smoke_test 1 \
    --num_episodes "$NUM_EPISODES" \
    --fps "$FPS" \
    --gpu_count "$GPU_COUNT" \
    --eval_time_set_day 0 \
    --viewer_rendering_disabled 1 \
    --gpu_mode egl

echo "=== [2/5] Collecting (waits for all episodes, then stops containers) ==="
python3 orchestrate.py start --compose-dir "$COMPOSE_DIR" --logs-dir "$LOGS_DIR"

echo "=== [3/5] Fixing docker-owned file permissions ==="
docker run --rm -v "$(pwd)/$BATCH_DIR:/workspace" alpine chown -R "$(id -u):$(id -g)" /workspace
docker run --rm -v "$(pwd)/$CACHE_BASE:/workspace" alpine chown -R "$(id -u):$(id -g)" /workspace

echo "=== [4/5] Aligning recordings (side + overhead, H.264) ==="
python3 orchestrate.py postprocess --compose-dir "$COMPOSE_DIR" --workers 32 --output-dir "$BATCH_DIR/aligned"

echo "=== [5/5] Flattening into dataset directory ==="
python3 postprocess/flatten_overhead_dataset.py \
    --source-dir "$BATCH_DIR" \
    --destination-dir "$DATASET_DIR" \
    --batch-name batch_0

echo ""
echo "Done. Dataset in: $DATASET_DIR"
ls "$DATASET_DIR"
echo "Tip: raw recordings in $BATCH_DIR/camera/output_* are large; delete them"
echo "     after verifying the dataset."

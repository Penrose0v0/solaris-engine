#!/bin/bash
# Trial collection of worldModEval (interaction fidelity / state propagation)
# episodes: one batch of normal-world episodes, aligned, flattened into
#   <OUT_ROOT>/dataset/worldModEval/{videos,trajectories,episode_info,first_frames}
# and checked with postprocess/world_mod_quality.py
#   (<OUT_ROOT>/dataset/quality_manifest_world_mod.json).
#
# Long run: launch with  setsid nohup ./run_world_mod_trial.sh > world_mod_trial.log 2>&1 &

OUT_ROOT="output/world_mod_trial001"
INSTANCES=2              # parallel normal worlds
EPISODES_PER_INSTANCE=10 # -> INSTANCES * EPISODES_PER_INSTANCE episodes
START_ID=0
GPU_COUNT=2

while [[ $# -gt 0 ]]; do
  case $1 in
    --output-dir)            OUT_ROOT="$2";              shift 2 ;;
    --instances)             INSTANCES="$2";             shift 2 ;;
    --episodes-per-instance) EPISODES_PER_INSTANCE="$2"; shift 2 ;;
    --start-id)              START_ID="$2";              shift 2 ;;
    --gpu-count)             GPU_COUNT="$2";             shift 2 ;;
    -h|--help)
      echo "Usage: $0 [OPTIONS]"
      echo "  --output-dir DIR             (default: $OUT_ROOT)"
      echo "  --instances N                worlds (default: $INSTANCES)"
      echo "  --episodes-per-instance N    (default: $EPISODES_PER_INSTANCE)"
      echo "  --start-id N                 first episode id (default: $START_ID)"
      echo "  --gpu-count N                (default: $GPU_COUNT)"
      exit 0
      ;;
    *) echo "Unknown option: $1 (use --help)" >&2; exit 1 ;;
  esac
done

source /storage/home/yang.hu/miniforge3/etc/profile.d/conda.sh
conda activate SolarisEngine
cd "$(dirname "$0")"

TYPE=worldModEval
DATASET="$OUT_ROOT/dataset"
PY_POST=.venv-postprocess/bin/python
BATCH_DIR="$OUT_ROOT/batch_0"
COMPOSE_DIR="$BATCH_DIR/compose_configs"

mkdir -p "$OUT_ROOT"
echo "trial started $(date -Is): ${INSTANCES}x${EPISODES_PER_INSTANCE} episodes from id $START_ID -> $BATCH_DIR"

docker ps -q --filter "name=docker-compose-" | xargs -r docker rm -f >/dev/null 2>&1

python3 generate_compose.py \
    --compose_dir "$COMPOSE_DIR" \
    --base_port 25590 --base_rcon_port 25600 \
    --act_recorder_port 8110 --coord_port 8120 \
    --data_dir "$BATCH_DIR/data" \
    --output_dir "$BATCH_DIR/output" \
    --camera_output_alpha_base "$BATCH_DIR/camera/output_alpha" \
    --camera_output_bravo_base "$BATCH_DIR/camera/output_bravo" \
    --camera_data_alpha_base "$BATCH_DIR/camera/data_alpha" \
    --camera_data_bravo_base "$BATCH_DIR/camera/data_bravo" \
    --smoke_test 0 \
    --num_flatland_world 0 \
    --num_normal_world "$INSTANCES" \
    --num_episodes "$EPISODES_PER_INSTANCE" \
    --episode_start_id "$START_ID" \
    --episode_types "$TYPE" \
    --viewer_rendering_disabled 1 \
    --gpu_mode egl \
    --gpu_count "$GPU_COUNT" \
    --eval_time_set_day 1 || { echo "generate_compose failed"; exit 1; }

python3 orchestrate.py start --build --compose-dir "$COMPOSE_DIR" --logs-dir "$BATCH_DIR/logs"
python3 orchestrate.py stop --compose-dir "$COMPOSE_DIR"
docker run --rm -v "$(pwd)/$BATCH_DIR:/workspace" alpine chown -R "$(id -u):$(id -g)" /workspace

n_raw=$(ls "$BATCH_DIR"/output/*_episode_info.json 2>/dev/null | wc -l)
echo "batch: $n_raw recordings produced"
if [[ "$n_raw" -eq 0 ]]; then
  echo "batch produced nothing; see $BATCH_DIR/logs. Stopping."
  exit 1
fi

PATH=$HOME/.local/bin:$PATH $PY_POST orchestrate.py postprocess \
    --compose-dir "$COMPOSE_DIR" --workers 32 --output-dir "$BATCH_DIR/aligned"
$PY_POST postprocess/flatten_overhead_dataset.py \
    --source-dir "$BATCH_DIR" --destination-dir "$DATASET/$TYPE" --batch-name "$TYPE"
$PY_POST postprocess/world_mod_quality.py --root "$DATASET"

echo "trial finished $(date -Is): recordings=$(ls "$DATASET/$TYPE/videos" 2>/dev/null | wc -l)"

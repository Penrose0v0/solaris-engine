#!/bin/bash
# Production collection of trajSegmentsEval (action-following eval) episodes.
#
# Runs batches of normal-world episodes until the dataset holds at least
# TARGET_PASSED recordings whose start-pose search succeeded
# (scene_check_passed), or MAX_EPISODES episodes have been recorded. Every
# batch continues the episode numbering (--episode_start_id) so file names
# and segment-script seeds never repeat across batches. Recordings that
# failed the scene check are kept and only flagged in quality_manifest.json.
#
# Output:
#   <OUT_ROOT>/batch_<n>/           raw per-batch collection (compose, logs, aligned)
#   <OUT_ROOT>/dataset/trajSegmentsEval/{videos,trajectories,episode_info,first_frames}
#   <OUT_ROOT>/dataset/quality_manifest.json
#
# Long run: launch with  setsid nohup ./run_traj_segments_prod.sh > prod.log 2>&1 &

OUT_ROOT="output/traj_segments_prod001"
INSTANCES=2            # parallel normal worlds per batch
EPISODES_PER_INSTANCE=20
TARGET_PASSED=160      # scene_check_passed recordings wanted
MAX_EPISODES=200       # hard cap on episodes (not recordings)
GPU_COUNT=2

while [[ $# -gt 0 ]]; do
  case $1 in
    --output-dir)            OUT_ROOT="$2";              shift 2 ;;
    --instances)             INSTANCES="$2";             shift 2 ;;
    --episodes-per-instance) EPISODES_PER_INSTANCE="$2"; shift 2 ;;
    --target-passed)         TARGET_PASSED="$2";         shift 2 ;;
    --max-episodes)          MAX_EPISODES="$2";          shift 2 ;;
    --gpu-count)             GPU_COUNT="$2";             shift 2 ;;
    -h|--help)
      echo "Usage: $0 [OPTIONS]"
      echo "  --output-dir DIR             (default: $OUT_ROOT)"
      echo "  --instances N                worlds per batch (default: $INSTANCES)"
      echo "  --episodes-per-instance N    (default: $EPISODES_PER_INSTANCE)"
      echo "  --target-passed N            stop once this many recordings passed the scene check (default: $TARGET_PASSED)"
      echo "  --max-episodes N             stop after this many episodes regardless (default: $MAX_EPISODES)"
      echo "  --gpu-count N                (default: $GPU_COUNT)"
      exit 0
      ;;
    *) echo "Unknown option: $1 (use --help)" >&2; exit 1 ;;
  esac
done

source /storage/home/yang.hu/miniforge3/etc/profile.d/conda.sh
conda activate SolarisEngine
cd "$(dirname "$0")"

TYPE=trajSegmentsEval
DATASET="$OUT_ROOT/dataset"
PY_POST=.venv-postprocess/bin/python
EPISODES_PER_BATCH=$((INSTANCES * EPISODES_PER_INSTANCE))

mkdir -p "$OUT_ROOT"
echo "prod started $(date -Is): target_passed=$TARGET_PASSED max_episodes=$MAX_EPISODES batch=${INSTANCES}x${EPISODES_PER_INSTANCE}"

count_passed() {
  [[ -f "$DATASET/quality_manifest.json" ]] || { echo 0; return; }
  python3 -c "import json,sys; print(sum(bool(r.get('scene_check_passed')) for r in json.load(open(sys.argv[1]))))" "$DATASET/quality_manifest.json"
}

episodes_done=0
batch=0
# Resume support: skip batches whose dataset entries already exist.
while [[ -d "$OUT_ROOT/batch_$batch/aligned" ]]; do
  episodes_done=$((episodes_done + EPISODES_PER_BATCH))
  batch=$((batch + 1))
done
[[ $batch -gt 0 ]] && echo "resuming after $batch batch(es), $episodes_done episodes"

while :; do
  passed=$(count_passed)
  echo "progress: episodes=$episodes_done scene_check_passed=$passed/$TARGET_PASSED"
  [[ $passed -ge $TARGET_PASSED ]] && { echo "target reached"; break; }
  [[ $episodes_done -ge $MAX_EPISODES ]] && { echo "episode cap reached without hitting target"; break; }

  BATCH_DIR="$OUT_ROOT/batch_$batch"
  COMPOSE_DIR="$BATCH_DIR/compose_configs"
  START_ID=$episodes_done
  echo "=========================================="
  echo "batch $batch: episodes $START_ID..$((START_ID + EPISODES_PER_BATCH - 1)) -> $BATCH_DIR  ($(date -Is))"
  echo "=========================================="

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

  BUILD=""; [[ $batch -eq 0 ]] && BUILD="--build"
  python3 orchestrate.py start $BUILD --compose-dir "$COMPOSE_DIR" --logs-dir "$BATCH_DIR/logs"
  python3 orchestrate.py stop --compose-dir "$COMPOSE_DIR"
  docker run --rm -v "$(pwd)/$BATCH_DIR:/workspace" alpine chown -R "$(id -u):$(id -g)" /workspace

  n_raw=$(ls "$BATCH_DIR"/output/*_episode_info.json 2>/dev/null | wc -l)
  echo "batch $batch: $n_raw recordings produced"
  if [[ "$n_raw" -eq 0 ]]; then
    echo "batch $batch produced nothing; see $BATCH_DIR/logs. Stopping."
    exit 1
  fi

  PATH=$HOME/.local/bin:$PATH $PY_POST orchestrate.py postprocess \
      --compose-dir "$COMPOSE_DIR" --workers 32 --output-dir "$BATCH_DIR/aligned"
  $PY_POST postprocess/flatten_overhead_dataset.py \
      --source-dir "$BATCH_DIR" --destination-dir "$DATASET/$TYPE" --batch-name "$TYPE"
  $PY_POST postprocess/traj_atomic_quality.py --root "$DATASET"

  episodes_done=$((episodes_done + EPISODES_PER_BATCH))
  batch=$((batch + 1))
done

echo "prod finished $(date -Is): episodes=$episodes_done recordings=$(ls "$DATASET/$TYPE/videos" | wc -l) scene_check_passed=$(count_passed)"

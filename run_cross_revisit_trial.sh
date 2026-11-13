#!/bin/bash
# Trial collection of crossRevisitEval (long-term shared-world consistency:
# in a courtyard the first observer turns to the north wall, looks, turns
# back; after a gap of 5 / 55 frames the second observer turns to it; control
# episodes: the first observer keeps looking): one batch, aligned, flattened
# into
#   <OUT_ROOT>/dataset/crossRevisitEval/{videos,trajectories,episode_info,first_frames}
# then annotated (phases, x_visibility) and checked with
# postprocess/cross_revisit_quality.py
#   (<OUT_ROOT>/dataset/quality_manifest_cross_revisit.json, review images in
#   <OUT_ROOT>/dataset/cross_revisit_review/).
#
# Long run: launch with  setsid nohup ./run_cross_revisit_trial.sh > cross_revisit_trial.log 2>&1 &

OUT_ROOT="output/cross_revisit_trial001"
DATASET=""              # flattened dataset dir (default: <OUT_ROOT>/dataset)
INSTANCES=2             # parallel normal worlds
EPISODES_PER_INSTANCE=4 # 2 instances x 4 episodes = the 8 variants (G 5/55 x main/control x B/A first) once each
START_ID=0
GPU_COUNT=2

while [[ $# -gt 0 ]]; do
  case $1 in
    --output-dir)            OUT_ROOT="$2";              shift 2 ;;
    --dataset-dir)           DATASET="$2";               shift 2 ;;
    --instances)             INSTANCES="$2";             shift 2 ;;
    --episodes-per-instance) EPISODES_PER_INSTANCE="$2"; shift 2 ;;
    --start-id)              START_ID="$2";              shift 2 ;;
    --gpu-count)             GPU_COUNT="$2";             shift 2 ;;
    -h|--help)
      echo "Usage: $0 [OPTIONS]"
      echo "  --output-dir DIR             raw batch (default: $OUT_ROOT)"
      echo "  --dataset-dir DIR            flattened dataset (default: <output-dir>/dataset)"
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

TYPE=crossRevisitEval
[[ -n "$DATASET" ]] || DATASET="$OUT_ROOT/dataset"
PY_POST=.venv-postprocess/bin/python
BATCH_DIR="$OUT_ROOT/batch_0"
COMPOSE_DIR="$BATCH_DIR/compose_configs"

mkdir -p "$OUT_ROOT" "$DATASET"
if [[ ! -f "$DATASET/README.md" ]]; then
  cp docs/cross_revisit_dataset_README.md "$DATASET/README.md"
  printf '\n## Collection\n\nRecorded with solaris-engine commit %s (branch %s), started %s, by %s.\n' \
    "$(git rev-parse HEAD)" "$(git rev-parse --abbrev-ref HEAD)" "$(date -Is)" "$(basename "$0")" >> "$DATASET/README.md"
fi
echo "trial started $(date -Is): ${INSTANCES}x${EPISODES_PER_INSTANCE} episodes from id $START_ID -> $BATCH_DIR"

docker ps -q --filter "name=docker-compose-" | xargs -r docker rm -f >/dev/null 2>&1

python3 generate_compose.py \
    --compose_dir "$COMPOSE_DIR" \
    --base_port 26590 --base_rcon_port 26600 \
    --act_recorder_port 9110 --coord_port 9120 \
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
$PY_POST postprocess/cross_revisit_quality.py --root "$DATASET"

echo "trial finished $(date -Is): recordings=$(ls "$DATASET/$TYPE/videos" 2>/dev/null | wc -l)"

#!/bin/bash
# Trial collection of sharedPersistEval (shared-world persistence with a
# conditioning frame: in the crossRevisitEval courtyard the first bot sees the
# north wall in the conditioning frame and turns away; after a gap of 5 / 55
# frames the other bot turns to it): one batch, aligned, flattened into
#   <OUT_ROOT>/dataset/sharedPersistEval/{videos,trajectories,episode_info,first_frames}
# then annotated (phases, condition_frame, x_visibility) and checked with
# postprocess/shared_persist_quality.py
#   (<OUT_ROOT>/dataset/quality_manifest_shared_persist.json, review images in
#   <OUT_ROOT>/dataset/shared_persist_review/).
#
# Long run: launch with  setsid nohup ./run_shared_persist_trial.sh > shared_persist_trial.log 2>&1 &

OUT_ROOT="output/shared_persist_trial001"
DATASET=""              # flattened dataset dir (default: <OUT_ROOT>/dataset)
INSTANCES=2             # parallel normal worlds
EPISODES_PER_INSTANCE=2 # 2 instances x 2 episodes = the 4 variants (G 5/55 x B/A first) once each; keep --start-id even
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

TYPE=sharedPersistEval
# Own compose project prefix, ports and X displays: this collection can run
# next to another one that uses the default "docker-compose-" projects.
export SOLARIS_COMPOSE_PREFIX=spersist
[[ -n "$DATASET" ]] || DATASET="$OUT_ROOT/dataset"
PY_POST=.venv-postprocess/bin/python
BATCH_DIR="$OUT_ROOT/batch_0"
COMPOSE_DIR="$BATCH_DIR/compose_configs"

mkdir -p "$OUT_ROOT" "$DATASET"
if [[ ! -f "$DATASET/README.md" ]]; then
  cp docs/shared_persist_dataset_README.md "$DATASET/README.md"
  printf '\n## Collection\n\nRecorded with solaris-engine commit %s (branch %s), started %s, by %s.\n' \
    "$(git rev-parse HEAD)" "$(git rev-parse --abbrev-ref HEAD)" "$(date -Is)" "$(basename "$0")" >> "$DATASET/README.md"
fi
echo "trial started $(date -Is): ${INSTANCES}x${EPISODES_PER_INSTANCE} episodes from id $START_ID -> $BATCH_DIR"

docker ps -aq --filter "name=${SOLARIS_COMPOSE_PREFIX}-" | xargs -r docker rm -f >/dev/null 2>&1

python3 generate_compose.py \
    --compose_dir "$COMPOSE_DIR" \
    --base_port 26690 --base_rcon_port 26700 \
    --act_recorder_port 9130 --coord_port 9140 \
    --display_base 299 \
    --camera_alpha_vnc_base 5951 --camera_alpha_novnc_base 6951 \
    --camera_bravo_vnc_base 5952 --camera_bravo_novnc_base 6952 \
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
$PY_POST postprocess/shared_persist_quality.py --root "$DATASET"

echo "trial finished $(date -Is): recordings=$(ls "$DATASET/$TYPE/videos" 2>/dev/null | wc -l)"

#!/bin/bash
# Trial collection of worldModRevisitEval ("EP": interact, yaw out 120 deg,
# hold H, yaw back; single agent, both bots record their own episode far
# apart) and worldModOffscreenEval ("DSP": A interacts while B looks 90 deg
# away, B turns onto the target after H frames). One batch per type, each
# covering its 12 variants (H 0/40/160 x place/break x left/right) once,
# aligned, flattened into
#   <OUT_ROOT>/dataset/<type>/{videos,trajectories,episode_info,first_frames}
# then annotated and checked with postprocess/world_mod_offscreen_quality.py
#   (<OUT_ROOT>/dataset/quality_manifest_world_mod_offscreen.json, review
#   images in <OUT_ROOT>/dataset/world_mod_offscreen_review/).
#
# Long run: launch with
#   setsid nohup ./run_world_mod_offscreen_trial.sh > world_mod_offscreen_trial.log 2>&1 &

OUT_ROOT="output/world_mod_offscreen_trial001"
DATASET=""
INSTANCES=2
START_ID=0
GPU_COUNT=2
TYPES="worldModRevisitEval worldModOffscreenEval"

while [[ $# -gt 0 ]]; do
  case $1 in
    --output-dir)  OUT_ROOT="$2";  shift 2 ;;
    --dataset-dir) DATASET="$2";   shift 2 ;;
    --instances)   INSTANCES="$2"; shift 2 ;;
    --start-id)    START_ID="$2";  shift 2 ;;
    --gpu-count)   GPU_COUNT="$2"; shift 2 ;;
    --types)       TYPES="$2";     shift 2 ;;
    -h|--help)
      echo "Usage: $0 [--output-dir DIR] [--dataset-dir DIR] [--instances N] [--start-id N] [--gpu-count N] [--types \"T1 T2\"]"
      exit 0 ;;
    *) echo "Unknown option: $1 (use --help)" >&2; exit 1 ;;
  esac
done

source /storage/home/yang.hu/miniforge3/etc/profile.d/conda.sh
conda activate SolarisEngine
cd "$(dirname "$0")"

[[ -n "$DATASET" ]] || DATASET="$OUT_ROOT/dataset"
PY_POST=.venv-postprocess/bin/python
mkdir -p "$OUT_ROOT" "$DATASET"
if [[ ! -f "$DATASET/README.md" && -f docs/world_mod_offscreen_dataset_README.md ]]; then
  cp docs/world_mod_offscreen_dataset_README.md "$DATASET/README.md"
  printf '\n## Collection\n\nRecorded with solaris-engine commit %s (branch %s, working tree may hold uncommitted changes), started %s, by %s.\n' \
    "$(git rev-parse HEAD)" "$(git rev-parse --abbrev-ref HEAD)" "$(date -Is)" "$(basename "$0")" >> "$DATASET/README.md"
fi

for TYPE in $TYPES; do
  # EP: 6 (H, interaction) cells x 2 bots (left / right) = 2 worlds x 3 episodes;
  # DSP: 12 cells = 2 worlds x 6 episodes.
  if [[ "$TYPE" == "worldModRevisitEval" ]]; then EPISODES=3; else EPISODES=6; fi
  BATCH_DIR="$OUT_ROOT/batch_$TYPE"
  COMPOSE_DIR="$BATCH_DIR/compose_configs"
  echo "[$TYPE] started $(date -Is): ${INSTANCES}x${EPISODES} episodes from id $START_ID -> $BATCH_DIR"

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
      --num_episodes "$EPISODES" \
      --episode_start_id "$START_ID" \
      --episode_types "$TYPE" \
      --viewer_rendering_disabled 1 \
      --gpu_mode egl \
      --gpu_count "$GPU_COUNT" \
      --eval_time_set_day 1 || { echo "generate_compose failed"; exit 1; }

  timeout 3000 python3 orchestrate.py start --build --compose-dir "$COMPOSE_DIR" --logs-dir "$BATCH_DIR/logs"
  python3 orchestrate.py stop --compose-dir "$COMPOSE_DIR"
  docker run --rm -v "$(pwd)/$BATCH_DIR:/workspace" alpine chown -R "$(id -u):$(id -g)" /workspace

  n_raw=$(ls "$BATCH_DIR"/output/*_episode_info.json 2>/dev/null | wc -l)
  echo "[$TYPE] batch: $n_raw recordings produced"
  if [[ "$n_raw" -eq 0 ]]; then
    echo "[$TYPE] batch produced nothing; see $BATCH_DIR/logs."
    continue
  fi
  PATH=$HOME/.local/bin:$PATH $PY_POST orchestrate.py postprocess \
      --compose-dir "$COMPOSE_DIR" --workers 32 --output-dir "$BATCH_DIR/aligned"
  $PY_POST postprocess/flatten_overhead_dataset.py \
      --source-dir "$BATCH_DIR" --destination-dir "$DATASET/$TYPE" --batch-name "$TYPE"
done

$PY_POST postprocess/world_mod_offscreen_quality.py --root "$DATASET"
echo "trial finished $(date -Is)"

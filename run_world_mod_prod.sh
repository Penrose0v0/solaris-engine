#!/bin/bash
# Production collection of worldModEval (interaction fidelity / state
# propagation) episodes.
#
# Runs batches of normal-world episodes until the dataset holds at least
# TARGET_CLEAN episodes that pass every check in
# postprocess/world_mod_quality.py, or MAX_EPISODES episodes have been
# recorded. Every batch continues the episode numbering (--episode_start_id)
# so file names and plan seeds never repeat across batches. The A/B view
# angle band (10-30 / 30-90 deg) is drawn per episode by the handler, half
# each; the quality summary reports the counts per band.
#
# Output:
#   <OUT_ROOT>/batch_<n>/           raw per-batch collection (compose, logs, aligned)
#   <DATASET>/worldModEval/{videos,trajectories,episode_info,first_frames}
#   <DATASET>/quality_manifest_world_mod.json
#
# Long run: launch with  setsid nohup ./run_world_mod_prod.sh > world_mod_prod.log 2>&1 &

OUT_ROOT="output/world_mod_prod001"
DATASET="/home/yang.hu/share/datasets/solaris-world-mod-prod001"
INSTANCES=2            # parallel normal worlds per batch
EPISODES_PER_INSTANCE=10
TARGET_CLEAN=90        # clean episodes wanted
MAX_EPISODES=130       # hard cap on episodes
GPU_COUNT=2
FORCE_BUILD=0          # rebuild the images for the first batch of this invocation

while [[ $# -gt 0 ]]; do
  case $1 in
    --build)                 FORCE_BUILD=1;              shift ;;
    --output-dir)            OUT_ROOT="$2";              shift 2 ;;
    --dataset-dir)           DATASET="$2";               shift 2 ;;
    --instances)             INSTANCES="$2";             shift 2 ;;
    --episodes-per-instance) EPISODES_PER_INSTANCE="$2"; shift 2 ;;
    --target-clean)          TARGET_CLEAN="$2";          shift 2 ;;
    --max-episodes)          MAX_EPISODES="$2";          shift 2 ;;
    --gpu-count)             GPU_COUNT="$2";             shift 2 ;;
    -h|--help)
      echo "Usage: $0 [OPTIONS]"
      echo "  --build                      rebuild the images before the first batch of this run (always done for batch 0)"
      echo "  --output-dir DIR             raw batches (default: $OUT_ROOT)"
      echo "  --dataset-dir DIR            flattened dataset (default: $DATASET)"
      echo "  --instances N                worlds per batch (default: $INSTANCES)"
      echo "  --episodes-per-instance N    (default: $EPISODES_PER_INSTANCE)"
      echo "  --target-clean N             stop once this many episodes are clean (default: $TARGET_CLEAN)"
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

TYPE=worldModEval
PY_POST=.venv-postprocess/bin/python
EPISODES_PER_BATCH=$((INSTANCES * EPISODES_PER_INSTANCE))

mkdir -p "$OUT_ROOT" "$DATASET"
echo "prod started $(date -Is): target_clean=$TARGET_CLEAN max_episodes=$MAX_EPISODES batch=${INSTANCES}x${EPISODES_PER_INSTANCE} dataset=$DATASET"
[[ -f "$DATASET/README.md" ]] || cp docs/world_mod_dataset_README.md "$DATASET/README.md"

count_clean() {
  [[ -f "$DATASET/quality_manifest_world_mod.json" ]] || { echo 0; return; }
  python3 -c "import json,sys; print(sum(bool(r.get('clean')) for r in json.load(open(sys.argv[1]))))" "$DATASET/quality_manifest_world_mod.json"
}

episodes_done=0
batch=0
# Resume support: skip batches whose aligned output already exists.
while [[ -d "$OUT_ROOT/batch_$batch/aligned" ]]; do
  episodes_done=$((episodes_done + EPISODES_PER_BATCH))
  batch=$((batch + 1))
done
[[ $batch -gt 0 ]] && echo "resuming after $batch batch(es), $episodes_done episodes"

while :; do
  clean=$(count_clean)
  echo "progress: episodes=$episodes_done clean=$clean/$TARGET_CLEAN"
  [[ $clean -ge $TARGET_CLEAN ]] && { echo "target reached"; break; }
  [[ $episodes_done -ge $MAX_EPISODES ]] && { echo "episode cap reached without hitting target"; break; }
  remaining=$((MAX_EPISODES - episodes_done))
  per_instance=$EPISODES_PER_INSTANCE
  if [[ $remaining -lt $EPISODES_PER_BATCH ]]; then
    per_instance=$(( (remaining + INSTANCES - 1) / INSTANCES ))
  fi

  BATCH_DIR="$OUT_ROOT/batch_$batch"
  COMPOSE_DIR="$BATCH_DIR/compose_configs"
  START_ID=$episodes_done
  echo "=========================================="
  echo "batch $batch: episodes $START_ID..$((START_ID + INSTANCES * per_instance - 1)) -> $BATCH_DIR  ($(date -Is))"
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
      --num_episodes "$per_instance" \
      --episode_start_id "$START_ID" \
      --episode_types "$TYPE" \
      --viewer_rendering_disabled 1 \
      --gpu_mode egl \
      --gpu_count "$GPU_COUNT" \
      --eval_time_set_day 1 || { echo "generate_compose failed"; exit 1; }

  BUILD=""; [[ $batch -eq 0 || $FORCE_BUILD -eq 1 ]] && BUILD="--build"
  FORCE_BUILD=0
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
  $PY_POST postprocess/world_mod_quality.py --root "$DATASET"
  $PY_POST postprocess/camera_fov_check.py --root "$DATASET" --out "$DATASET/camera_fov_check.json" | tail -3

  # Every camera must render at the nominal FOV 70 deg: stop on the first
  # recording that either check flags as boosted (77 deg).
  boosted=$(python3 - "$DATASET" <<'PYEOF'
import json, sys
root = sys.argv[1]
bad = set()
for r in json.load(open(f"{root}/quality_manifest_world_mod.json")):
    for side in ("actor_recording", "observer_recording"):
        if r.get(side) and r[side].get("fov_boosted"):
            bad.add(f"{r['episode']}/{r['instance']}/{r[side]['bot']} (silhouette scale {r[side]['fov_scale']:.2f})")
for name, v in json.load(open(f"{root}/camera_fov_check.json")).items():
    if v.get("verdict") == "boosted":
        bad.add(f"{name} (f_est {v['f_est']:.0f})")
print("\n".join(sorted(bad)))
PYEOF
)
  if [[ -n "$boosted" ]]; then
    echo "FOV CHECK FAILED after batch $batch: recordings rendered at FOV 77 deg:"
    echo "$boosted"
    echo "stopping production"
    exit 2
  fi

  episodes_done=$((episodes_done + INSTANCES * per_instance))
  batch=$((batch + 1))
done

echo "prod finished $(date -Is): episodes=$episodes_done recordings=$(ls "$DATASET/$TYPE/videos" 2>/dev/null | wc -l) clean=$(count_clean)"

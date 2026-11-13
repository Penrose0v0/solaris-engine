#!/bin/bash
# Production collection of crossRevisitEval (long-term shared-world
# consistency by cross revisit: in a courtyard the first observer turns to
# the north wall, looks, turns back; after a gap of 5 / 55 frames the second
# observer turns to it; control episodes: the first observer keeps looking).
#
# Runs batches of normal-world episodes until the dataset holds at least
# TARGET_CLEAN clean episodes (both recordings clean) in EACH of the four
# cells main / control x G5 / G55 (the 8 variants, incl. who looks first,
# cycle with 4 * instance + episode) according to
# postprocess/cross_revisit_quality.py, or MAX_EPISODES episodes have been
# recorded. Every batch continues the episode numbering (--episode_start_id)
# so file names and plan seeds never repeat.
#
# Output:
#   <OUT_ROOT>/batch_<n>/           raw per-batch collection (compose, logs, aligned)
#   <DATASET>/crossRevisitEval/{videos,trajectories,episode_info,first_frames}
#   <DATASET>/quality_manifest_cross_revisit.json, <DATASET>/cross_revisit_review/
#
# Long run: launch with  setsid nohup ./run_cross_revisit_prod.sh > cross_revisit_prod.log 2>&1 &

OUT_ROOT="output/cross_revisit_prod001"
DATASET="/home/yang.hu/share/datasets/solaris-cross-revisit-prod001"
INSTANCES=2            # parallel normal worlds per batch
EPISODES_PER_INSTANCE=16 # a multiple of 8: every instance cycles through the 8 variants
TARGET_CLEAN=30        # clean episodes wanted per group x G (main/control x G5/G55)
MAX_EPISODES=192       # hard cap on episodes
GPU_COUNT=2
FORCE_BUILD=0          # rebuild the images for the first batch of this invocation
BATCH_TIMEOUT=3000     # seconds per batch before a hung batch is cut off 

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
      echo "  --target-clean N             stop once this many episodes are clean in each group x G cell (default: $TARGET_CLEAN)"
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

TYPE=crossRevisitEval
PY_POST=.venv-postprocess/bin/python
EPISODES_PER_BATCH=$((INSTANCES * EPISODES_PER_INSTANCE))

mkdir -p "$OUT_ROOT" "$DATASET"
echo "prod started $(date -Is): target_clean=$TARGET_CLEAN max_episodes=$MAX_EPISODES batch=${INSTANCES}x${EPISODES_PER_INSTANCE} dataset=$DATASET"
if [[ ! -f "$DATASET/README.md" ]]; then
  cp docs/cross_revisit_dataset_README.md "$DATASET/README.md"
  printf '\n## Collection\n\nRecorded with solaris-engine commit %s (branch %s), started %s, by %s.\n' \
    "$(git rev-parse HEAD)" "$(git rev-parse --abbrev-ref HEAD)" "$(date -Is)" "$(basename "$0")" >> "$DATASET/README.md"
fi

MANIFEST="$DATASET/quality_manifest_cross_revisit.json"
# Clean episodes (one record per episode) in the cell with the fewest.
CELLS_PY='import json,sys,collections
r=json.load(open(sys.argv[1]))
eps={(x["instance"],x["episode"]):x for x in r}
cells=[(g,G) for g in ("main","control") for G in (5,55)]
tot=collections.Counter((x["group"],x["gap_frames"]) for x in eps.values())
ok=collections.Counter((x["group"],x["gap_frames"]) for x in eps.values() if x.get("episode_clean"))
order=collections.Counter((x["group"],x["gap_frames"],x["first_observer_agent"]) for x in eps.values() if x.get("episode_clean"))'
count_clean() {
  [[ -f "$MANIFEST" ]] || { echo 0; return; }
  python3 -c "$CELLS_PY
print(min(ok[c] for c in cells))" "$MANIFEST"
}
clean_by_type() {
  [[ -f "$MANIFEST" ]] || { echo "none"; return; }
  python3 -c "$CELLS_PY
print(' '.join(f'{g}/G{G}: {ok[(g,G)]} clean / {tot[(g,G)]} (A-first {order[(g,G,\"A\")]}, B-first {order[(g,G,\"B\")]})' for g,G in cells))" "$MANIFEST"
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
  echo "progress: episodes=$episodes_done min clean per cell=$clean/$TARGET_CLEAN ($(clean_by_type))"
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
      --num_episodes "$per_instance" \
      --episode_start_id "$START_ID" \
      --episode_types "$TYPE" \
      --viewer_rendering_disabled 1 \
      --gpu_mode egl \
      --gpu_count "$GPU_COUNT" \
      --eval_time_set_day 1 || { echo "generate_compose failed"; exit 1; }

  BUILD=""; [[ $batch -eq 0 || $FORCE_BUILD -eq 1 ]] && BUILD="--build"
  FORCE_BUILD=0
  # A bot that gets disconnected mid-batch (ECONNRESET / "exceeding packet
  # rate limit") leaves its controller idle and `start` never returns; cap
  # the batch and keep whatever was recorded.
  timeout --signal=INT "$BATCH_TIMEOUT" python3 orchestrate.py start $BUILD --compose-dir "$COMPOSE_DIR" --logs-dir "$BATCH_DIR/logs" \
    || echo "batch $batch: orchestrate start ended with status $? (timeout ${BATCH_TIMEOUT}s or error), continuing with what was recorded"
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
  $PY_POST postprocess/cross_revisit_quality.py --root "$DATASET" | tail -8

  episodes_done=$((episodes_done + INSTANCES * per_instance))
  batch=$((batch + 1))
done

echo "prod finished $(date -Is): episodes=$episodes_done recordings=$(ls "$DATASET/$TYPE/videos" 2>/dev/null | wc -l) $(clean_by_type)"

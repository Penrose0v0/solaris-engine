#!/bin/bash
# Production collection of worldModRevisitEval ("EP") and
# worldModOffscreenEval ("DSP"); see docs/world_mod_offscreen_dataset_README.md.
#
# For each type, runs batches of normal-world episodes until, according to
# postprocess/world_mod_offscreen_quality.py, every hold tier H = 0 / 40 / 160
# has at least TARGET_CLEAN clean units (EP: recordings; DSP: episodes, both
# recordings clean), with at least TARGET_CLEAN / 2 in each half of place /
# break and of left / right, or the type's episode cap is reached. Every
# batch continues the episode numbering (--episode_start_id): the variants
# cycle with the episode number, file names and seeds never repeat.
#
# Output:
#   <OUT_ROOT>/batch_<type>_<n>/    raw per-batch collection (compose, logs, aligned)
#   <DATASET>/<type>/{videos,trajectories,episode_info,first_frames}
#   <DATASET>/quality_manifest_world_mod_offscreen.json, <DATASET>/world_mod_offscreen_review/
#
# Long run: launch with
#   setsid nohup ./run_world_mod_offscreen_prod.sh > world_mod_offscreen_prod.log 2>&1 &

OUT_ROOT="output/world_mod_offscreen_prod001"
DATASET="/home/yang.hu/share/datasets/solaris-world-mod-offscreen-prod001"
INSTANCES=2
EPISODES_PER_INSTANCE=12  # a multiple of 6 (EP) and 12 (DSP): every instance cycles through all variants
TARGET_CLEAN=30           # clean units wanted per H
MAX_EPISODES_EP=120       # hard caps on episodes (EP: 2 recordings each)
MAX_EPISODES_DSP=240
GPU_COUNT=2
FORCE_BUILD=0
BATCH_TIMEOUT=3000
TYPES="worldModRevisitEval worldModOffscreenEval"

while [[ $# -gt 0 ]]; do
  case $1 in
    --build)                 FORCE_BUILD=1;              shift ;;
    --output-dir)            OUT_ROOT="$2";              shift 2 ;;
    --dataset-dir)           DATASET="$2";               shift 2 ;;
    --instances)             INSTANCES="$2";             shift 2 ;;
    --episodes-per-instance) EPISODES_PER_INSTANCE="$2"; shift 2 ;;
    --target-clean)          TARGET_CLEAN="$2";          shift 2 ;;
    --max-episodes-ep)       MAX_EPISODES_EP="$2";       shift 2 ;;
    --max-episodes-dsp)      MAX_EPISODES_DSP="$2";      shift 2 ;;
    --gpu-count)             GPU_COUNT="$2";             shift 2 ;;
    --types)                 TYPES="$2";                 shift 2 ;;
    -h|--help)
      echo "Usage: $0 [--build] [--output-dir DIR] [--dataset-dir DIR] [--instances N] [--episodes-per-instance N]"
      echo "          [--target-clean N] [--max-episodes-ep N] [--max-episodes-dsp N] [--gpu-count N] [--types \"T1 T2\"]"
      exit 0 ;;
    *) echo "Unknown option: $1 (use --help)" >&2; exit 1 ;;
  esac
done

source /storage/home/yang.hu/miniforge3/etc/profile.d/conda.sh
conda activate SolarisEngine
cd "$(dirname "$0")"

PY_POST=.venv-postprocess/bin/python
EPISODES_PER_BATCH=$((INSTANCES * EPISODES_PER_INSTANCE))
MANIFEST="$DATASET/quality_manifest_world_mod_offscreen.json"

mkdir -p "$OUT_ROOT" "$DATASET"
echo "prod started $(date -Is): target_clean=$TARGET_CLEAN per H, batch=${INSTANCES}x${EPISODES_PER_INSTANCE}, dataset=$DATASET"
if [[ ! -f "$DATASET/README.md" ]]; then
  cp docs/world_mod_offscreen_dataset_README.md "$DATASET/README.md"
  printf '\n## Collection\n\nRecorded with solaris-engine commit %s (branch %s), started %s, by %s.\n' \
    "$(git rev-parse HEAD)" "$(git rev-parse --abbrev-ref HEAD)" "$(date -Is)" "$(basename "$0")" >> "$DATASET/README.md"
fi

# Units: EP one per recording, DSP one per episode (episode_clean).
STATS_PY='import json,sys,collections
t,target=sys.argv[2],int(sys.argv[3])
r=[x for x in json.load(open(sys.argv[1])) if x["type"]==t]
units={}
for x in r:
    key=(x["instance"],x["episode"],x["bot"] if x["eval_kind"]=="EP" else "")
    units.setdefault(key,x)
ok=[x for x in units.values() if x.get("episode_clean")]
H=(0,40,160)
byH=collections.Counter(x["hold_frames"] for x in ok)
half=collections.Counter()
for x in ok:
    half[(x["hold_frames"],x["interaction"])]+=1; half[(x["hold_frames"],x["direction"])]+=1
need=(target+1)//2
done=all(byH[h]>=target for h in H) and all(half[(h,v)]>=need for h in H for v in ("place","break","left","right"))'
is_done() {
  [[ -f "$MANIFEST" ]] || { echo 0; return; }
  python3 -c "$STATS_PY
print(1 if done else 0)" "$MANIFEST" "$1" "$TARGET_CLEAN"
}
stats() {
  [[ -f "$MANIFEST" ]] || { echo "none"; return; }
  python3 -c "$STATS_PY
print(f'units {len(units)}, clean {len(ok)}: ' + '; '.join(f'H{h}: {byH[h]} (place {half[(h,\"place\")]} / break {half[(h,\"break\")]}, left {half[(h,\"left\")]} / right {half[(h,\"right\")]})' for h in H))" "$MANIFEST" "$1" "$TARGET_CLEAN"
}

first_build=1
for TYPE in $TYPES; do
  if [[ "$TYPE" == "worldModRevisitEval" ]]; then MAX_EPISODES=$MAX_EPISODES_EP; else MAX_EPISODES=$MAX_EPISODES_DSP; fi
  episodes_done=0
  batch=0
  # Resume support: skip batches whose aligned output already exists.
  while [[ -d "$OUT_ROOT/batch_${TYPE}_$batch/aligned" ]]; do
    episodes_done=$((episodes_done + EPISODES_PER_BATCH))
    batch=$((batch + 1))
  done
  [[ $batch -gt 0 ]] && echo "[$TYPE] resuming after $batch batch(es), $episodes_done episodes"

  while :; do
    echo "[$TYPE] progress: episodes=$episodes_done; $(stats "$TYPE")"
    [[ $(is_done "$TYPE") -eq 1 ]] && { echo "[$TYPE] target reached"; break; }
    [[ $episodes_done -ge $MAX_EPISODES ]] && { echo "[$TYPE] episode cap reached without hitting target"; break; }

    BATCH_DIR="$OUT_ROOT/batch_${TYPE}_$batch"
    COMPOSE_DIR="$BATCH_DIR/compose_configs"
    START_ID=$episodes_done
    echo "=========================================="
    echo "[$TYPE] batch $batch: episodes $START_ID..$((START_ID + EPISODES_PER_BATCH - 1)) -> $BATCH_DIR  ($(date -Is))"
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
        --num_episodes "$EPISODES_PER_INSTANCE" \
        --episode_start_id "$START_ID" \
        --episode_types "$TYPE" \
        --viewer_rendering_disabled 1 \
        --gpu_mode egl \
        --gpu_count "$GPU_COUNT" \
        --eval_time_set_day 1 || { echo "generate_compose failed"; exit 1; }

    BUILD=""; [[ $first_build -eq 1 || $FORCE_BUILD -eq 1 ]] && BUILD="--build"
    first_build=0; FORCE_BUILD=0
    # A bot disconnected mid-batch leaves its controller idle and `start`
    # never returns; cap the batch and keep whatever was recorded.
    timeout --signal=INT "$BATCH_TIMEOUT" python3 orchestrate.py start $BUILD --compose-dir "$COMPOSE_DIR" --logs-dir "$BATCH_DIR/logs" \
      || echo "[$TYPE] batch $batch: orchestrate start ended with status $? (timeout ${BATCH_TIMEOUT}s or error), continuing with what was recorded"
    python3 orchestrate.py stop --compose-dir "$COMPOSE_DIR"
    docker run --rm -v "$(pwd)/$BATCH_DIR:/workspace" alpine chown -R "$(id -u):$(id -g)" /workspace

    n_raw=$(ls "$BATCH_DIR"/output/*_episode_info.json 2>/dev/null | wc -l)
    echo "[$TYPE] batch $batch: $n_raw recordings produced"
    if [[ "$n_raw" -eq 0 ]]; then
      echo "[$TYPE] batch $batch produced nothing; see $BATCH_DIR/logs. Stopping."
      exit 1
    fi

    PATH=$HOME/.local/bin:$PATH $PY_POST orchestrate.py postprocess \
        --compose-dir "$COMPOSE_DIR" --workers 32 --output-dir "$BATCH_DIR/aligned"
    $PY_POST postprocess/flatten_overhead_dataset.py \
        --source-dir "$BATCH_DIR" --destination-dir "$DATASET/$TYPE" --batch-name "$TYPE"
    $PY_POST postprocess/world_mod_offscreen_quality.py --root "$DATASET" | tail -3 | cut -c1-300

    episodes_done=$((episodes_done + EPISODES_PER_BATCH))
    batch=$((batch + 1))
  done
done

for TYPE in $TYPES; do echo "[$TYPE] final: $(stats "$TYPE")"; done
echo "prod finished $(date -Is)"

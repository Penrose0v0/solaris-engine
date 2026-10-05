#!/bin/bash
# GT production: chase / orbit / buildStructure, in batches. Each batch runs
# the three types (normal world, overhead cams, per-type dirs) and is copied to
# the shared datasets dir as solaris-prod-<batch>/ as soon as it completes.
#
# Per-batch quantities = INSTANCES x episodes/instance:
#   chase          4 x 8  = 32   (target N=60  -> 2 batches ~ 64)
#   orbit          4 x 12 = 48   (target N*1.5 -> 2 batches ~ 96)
#   buildStructure 4 x 16 = 64   (target N*2   -> 2 batches ~ 128)

BATCHES="${BATCHES:-1 2}"
INSTANCES="${INSTANCES:-4}"
GPU_COUNT="${GPU_COUNT:-2}"
SHARE_ROOT="${SHARE_ROOT:-/home/yang.hu/share/datasets}"
declare -A EPS=( [chase]=8 [orbit]=12 [buildStructure]=16 )

cd "$(dirname "$0")"
for b in $BATCHES; do
  tag=$(printf "%03d" "$b")
  OUT="output/prod_$tag"
  echo "################ batch $tag -> $OUT"
  for t in chase orbit buildStructure; do
    ./run_trial_by_type.sh --types "$t" --instances "$INSTANCES" --num-episodes "${EPS[$t]}" \
        --gpu-count "$GPU_COUNT" --output-dir "$OUT"
  done
  # copy this batch to share
  DEST="$SHARE_ROOT/solaris-prod-$tag"
  mkdir -p "$DEST"
  for t in chase orbit buildStructure; do
    [ -d "$OUT/$t/dataset" ] && cp -r "$OUT/$t/dataset" "$DEST/$t"
  done
  cp "$OUT/trial_summary.txt" "$DEST/summary.txt" 2>/dev/null
  echo "batch $tag copied to $DEST"
done
echo "ALL BATCHES DONE"

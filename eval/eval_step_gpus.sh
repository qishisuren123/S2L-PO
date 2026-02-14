#!/usr/bin/env bash
set -euo pipefail


if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <checkpoint_directory>"
  exit 1
fi

CHECKPOINT_DIR="$1"
if [ ! -d "$CHECKPOINT_DIR" ]; then
  echo "Checkpoint directory does not exist: $CHECKPOINT_DIR"
  exit 1
fi

cd /mnt/bn/protenix-3/xuyiran/yr/AnyMath

MODE="${MODE:-nothink}"
K="${K:-8}"
BENCHS="${BENCHS:-aime25 aime24}"
TP="${TP:-1}"
GPU_UTIL="${GPU_UTIL:-0.90}"
TS="${TS:-$(date +%y%m%d%H%M%S)}"
FIXED_TS="$TS"

NGPU="${NGPU:-8}"
GPU_IDS="${GPU_IDS:-}"
LOG_DIR="${LOG_DIR:-/eval/output_logs/eval_${TS}}"

mkdir -p "$LOG_DIR"

GPU_LIST_STR=""
if [ -n "$GPU_IDS" ]; then
  GPU_LIST_STR="$GPU_IDS"
elif [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  GPU_LIST_STR="${CUDA_VISIBLE_DEVICES}"
else
  for ((i=0; i<NGPU; i++)); do
    if [ -z "$GPU_LIST_STR" ]; then
      GPU_LIST_STR="$i"
    else
      GPU_LIST_STR="$GPU_LIST_STR,$i"
    fi
  done
fi

IFS=',' read -r -a GPU_LIST <<< "$GPU_LIST_STR"
TOTAL_GPU=${#GPU_LIST[@]}

if [ "$TP" -le 0 ]; then
  echo "Invalid TP: $TP"
  exit 1
fi

SLOTS=$(( TOTAL_GPU / TP ))
if [ "$SLOTS" -lt 1 ]; then
  echo "Not enough GPUs. TOTAL_GPU=$TOTAL_GPU TP=$TP"
  exit 1
fi

echo "GPU pool: [$GPU_LIST_STR] | TP=$TP | parallel slots=$SLOTS"
echo "Logs: $LOG_DIR"

fifo_path=$(mktemp -u)
mkfifo "$fifo_path"
cleanup() {
  rm -f "$fifo_path" 2>/dev/null || true
}
trap cleanup EXIT

exec 9<>"$fifo_path"
rm -f "$fifo_path"

for ((s=0; s<SLOTS; s++)); do
  echo "$s" >&9
done

declare -a PIDS=()
STRATEGY_NAME=$(basename "$CHECKPOINT_DIR")

for STEP_DIR in $(ls -d "$CHECKPOINT_DIR"/*/ | sort -V); do
  STEP_NAME=$(basename "$STEP_DIR")
  MODEL_DIR="$STEP_DIR${STRATEGY_NAME}_${STEP_NAME}"
  if [ ! -d "$MODEL_DIR" ]; then
    echo "Skip step $STEP_NAME: expected model dir not found: $MODEL_DIR"
    continue
  fi

  read -r slot <&9
  start=$(( slot * TP ))

  CUDA_DEVICES="${GPU_LIST[$start]}"
  for ((j=1; j<TP; j++)); do
    CUDA_DEVICES="${CUDA_DEVICES},${GPU_LIST[$((start+j))]}"
  done

  log_file="$LOG_DIR/${STRATEGY_NAME}_${STEP_NAME}.log"
  echo "[slot=$slot gpu=$CUDA_DEVICES] Running: $MODEL_DIR"

  (
    set +e
    export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"
    python run_evaluation.py \
      --model_path "$MODEL_DIR" \
      --mode "$MODE" \
      --k "$K" \
      --benchmarks $BENCHS \
      --tensor_parallel_size "$TP" \
      --gpu_memory_utilization "$GPU_UTIL" \
      --timestamp "$FIXED_TS" \
      >"$log_file" 2>&1
    rc=$?
    echo "[slot=$slot gpu=$CUDA_DEVICES] rc=$rc model=$MODEL_DIR log=$log_file" >> "$LOG_DIR/summary.txt"
    echo "$slot" >&9
    exit $rc
  ) &

  PIDS+=("$!")
done

failed=0
set +e
for pid in "${PIDS[@]}"; do
  wait "$pid"
  rc=$?
  if [ $rc -ne 0 ]; then
    failed=1
  fi
done
set -e

if [ $failed -ne 0 ]; then
  echo "Some evaluations failed. See $LOG_DIR/summary.txt and per-step logs in $LOG_DIR"
  exit 1
fi

echo "All evaluations completed successfully. Logs in $LOG_DIR"
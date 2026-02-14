#!/usr/bin/env bash
set -euo pipefail

# 检查输入参数
if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <checkpoint_directory>"
  exit 1
fi

CHECKPOINT_DIR="$1"
if [ ! -d "$CHECKPOINT_DIR" ]; then
  echo "Checkpoint directory does not exist: $CHECKPOINT_DIR"
  exit 1
fi

cd /eval


# MODEL="${MODEL:-/Qwen3-8B-Base}"
MODE="${MODE:-nothink}"                 # think | nothink
K="${K:-8}"
BENCHS="${BENCHS:-aime25 aime24}"
TP="${TP:-1}"                           # tensor_parallel_size
GPU_UTIL="${GPU_UTIL:-0.90}"            # gpu_memory_utilization
TS="${TS:-$(date +%y%m%d%H%M%S)}"


FIXED_TS="$TS"


# for STEP_DIR in $(ls -d "$CHECKPOINT_DIR"/*/ | sort -V | tac); do
for STEP_DIR in $(ls -d "$CHECKPOINT_DIR"/*/ | sort -V); do
# for STEP_DIR in "$CHECKPOINT_DIR"/*/; do
  STEP_NAME=$(basename "$STEP_DIR")
  echo "Running evaluation for step: $STEP_NAME"

  STRATEGY_NAME=$(basename "$CHECKPOINT_DIR")
  MODEL_DIR="$STEP_DIR${STRATEGY_NAME}_${STEP_NAME}"
  if [ ! -d "$MODEL_DIR" ]; then
    echo "Skip step $STEP_NAME: expected model dir not found: $MODEL_DIR"
    continue
  fi

  python run_evaluation.py \
    --model_path "$MODEL_DIR" \
    --mode "$MODE" \
    --k "$K" \
    --benchmarks $BENCHS \
    --tensor_parallel_size "$TP" \
    --gpu_memory_utilization "$GPU_UTIL" \
    --timestamp "$FIXED_TS"

    echo "$MODEL_DIR"
done

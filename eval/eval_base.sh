#!/usr/bin/env bash
set -euo pipefail

cd /eval

# -------------------------
# Common config (env override)
# -------------------------
MODEL="${MODEL:-/Qwen3-8B-Base}"
MODE="${MODE:-nothink}"                 # think | nothink
K="${K:-256}"
BENCHS="${BENCHS:-aime25 aime24}"
TP="${TP:-1}"                           # tensor_parallel_size
GPU_UTIL="${GPU_UTIL:-0.90}"            # gpu_memory_utilization
TS="${TS:-$(date +%y%m%d%H%M%S)}"

PARALLEL_8="${PARALLEL_8:-0}"          

# -------------------------
# 8-GPU parallel mode
# -------------------------
if [[ "$PARALLEL_8" == "1" ]]; then
  QWEN_WEIGHT_DIR="${QWEN_WEIGHT_DIR:-/qwen_weight}"

  MODEL_NAMES=(
    "Qwen3-1.7B-Base"
    "Qwen3-4B-Base"
    "Qwen3-8B-Base"
    "Qwen3-14B-Base"
  )

  MODEL_PATHS=(
    "$QWEN_WEIGHT_DIR/Qwen3-1.7B-Base"
    "$QWEN_WEIGHT_DIR/Qwen3-4B-Base"
    "$QWEN_WEIGHT_DIR/Qwen3-8B-Base"
    "$QWEN_WEIGHT_DIR/Qwen3-14B-Base"
  )

  MODES=("think" "nothink")

  GPUS=(0 1 2 3 4 5 6 7)


  for mp in "${MODEL_PATHS[@]}"; do
    if [[ ! -d "$mp" ]]; then
      echo "Model path not found: $mp" >&2
      exit 2
    fi
  done

  LOG_DIR="output/parallel_${TS}"
  mkdir -p "$LOG_DIR"

  cleanup() {
    echo "[cleanup] killing child processes..." >&2
    kill 0 || true
  }
  trap cleanup INT TERM

  pids=()
  descs=()

  task_i=0
  for mi in "${!MODEL_PATHS[@]}"; do
    for mode in "${MODES[@]}"; do
      gpu="${GPUS[$task_i]}"
      model_path="${MODEL_PATHS[$mi]}"
      model_name="${MODEL_NAMES[$mi]}"

      ts_i="$(date -d "+${task_i} seconds" +%y%m%d%H%M%S)"

      log="$LOG_DIR/job_${task_i}_${model_name}_${mode}_gpu${gpu}.log"

      echo "[launch] job=$task_i gpu=$gpu model=$model_name mode=$mode ts=$ts_i"
      echo "[log] $log"

      (
        set -euo pipefail
        export CUDA_VISIBLE_DEVICES="$gpu"
        python run_evaluation.py \
          --model_path "$model_path" \
          --mode "$mode" \
          --k "$K" \
          --benchmarks $BENCHS \
          --tensor_parallel_size 1 \
          --gpu_memory_utilization "$GPU_UTIL" \
          --timestamp "$ts_i"
      ) >"$log" 2>&1 &

      pids+=("$!")
      descs+=("job=$task_i gpu=$gpu model=$model_name mode=$mode ts=$ts_i log=$log")
      task_i=$((task_i + 1))
    done
  done

  fail=0
  for i in "${!pids[@]}"; do
    pid="${pids[$i]}"
    if ! wait "$pid"; then
      echo "[fail] ${descs[$i]}" >&2
      fail=1
    else
      echo "[done] ${descs[$i]}"
    fi
  done

  echo "[all done] logs: $LOG_DIR"
  exit "$fail"
fi

# -------------------------
# Default single-run mode (original behavior)
# -------------------------
python run_evaluation.py \
  --model_path "$MODEL" \
  --mode "$MODE" \
  --k "$K" \
  --benchmarks $BENCHS \
  --tensor_parallel_size "$TP" \
  --gpu_memory_utilization "$GPU_UTIL" \
  --timestamp "$TS"
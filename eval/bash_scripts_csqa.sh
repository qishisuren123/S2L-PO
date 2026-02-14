#!/usr/bin/env bash
# CommonsenseQA :bash run_eval_csqa.sh

set -euo pipefail
cd eval


MODEL="${MODEL:-/Qwen3-8B}"
MODE="${MODE:-nothink}"                 # think | nothink
K="${K:-8}"
BENCHS="${BENCHS:-csqa}"               # csqa, csqa_dev, csqa_test
TP="${TP:-1}"                          # tensor_parallel_size
GPU_UTIL="${GPU_UTIL:-0.90}"           # gpu_memory_utilization
TS="${TS:-$(date +%y%m%d%H%M%S)}"


python run_evaluation_csqa.py \
  --model_path "$MODEL" \
  --mode "$MODE" \
  --k "$K" \
  --benchmarks $BENCHS \
  --tensor_parallel_size "$TP" \
  --gpu_memory_utilization "$GPU_UTIL" \
  --timestamp "$TS"
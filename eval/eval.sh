#!/usr/bin/env bash

set -euo pipefail
cd /eval

MODEL="${MODEL:-/Qwen3-8B-Base}"
MODE="${MODE:-nothink}"                 # think | nothink
K="${K:-256}"
BENCHS="${BENCHS:-aime25 aime24}"
# BENCHS="${BENCHS:-aime25}"
TP="${TP:-1}"                           #  tensor_parallel_size
GPU_UTIL="${GPU_UTIL:-0.90}"            # gpu_memory_utilization
TS="${TS:-$(date +%y%m%d%H%M%S)}"


python run_evaluation.py \
  --model_path "$MODEL" \
  --mode "$MODE" \
  --k "$K" \
  --benchmarks $BENCHS \
  --tensor_parallel_size "$TP" \
  --gpu_memory_utilization "$GPU_UTIL" \
  --timestamp "$TS"
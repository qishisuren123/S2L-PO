#!/bin/bash
# Evaluate all three S2L-PO Qwen3 checkpoints on AIME24, AIME25, and MATH500.
#
# Usage:
#   # Edit MODEL_BASE below to point to your local release directory, then run:
#   bash eval_s2l_po.sh
#
# Each model runs on one GPU in parallel (requires ≥3 GPUs with ~20GB VRAM each
# for the 8B models, and ~30GB for the 14B model).
# Adjust CUDA_VISIBLE_DEVICES and --tensor_parallel_size as needed.

set -uo pipefail

# ── user-configurable paths ────────────────────────────────────────────────────
MODEL_BASE="${1:-/path/to/release}"   # root dir containing all three model folders
K="${2:-32}"                          # samples per question (Pass@k)
MODE="${3:-nothink}"                  # think | nothink

MODELS=(
  "Qwen3-8B-S2L-PO-1.7Bexplorer"
  "Qwen3-8B-S2L-PO-4Bexplorer"
  "Qwen3-14B-S2L-PO-4Bexplorer"
)
# GPU assignment: one per model (change as needed)
GPUS=(0 1 2)
# 14B needs 2 GPUs for tensor parallelism at k=32; set accordingly
TP=(1 1 1)
# ──────────────────────────────────────────────────────────────────────────────

cd "$(dirname "$0")"

TS=$(date +%y%m%d%H%M%S)
LOG_DIR="logs/eval_${TS}"
mkdir -p "$LOG_DIR"

echo "=================================================="
echo "S2L-PO Evaluation  mode=${MODE}  k=${K}"
echo "Timestamp: ${TS}  |  Logs: ${LOG_DIR}"
echo "=================================================="

declare -a PIDS=()

for i in "${!MODELS[@]}"; do
    MODEL="${MODELS[$i]}"
    GPU="${GPUS[$i]}"
    TP_SIZE="${TP[$i]}"
    MODEL_PATH="${MODEL_BASE}/${MODEL}"
    LOG="${LOG_DIR}/${MODEL}.log"

    (
        set +e
        export CUDA_VISIBLE_DEVICES="${GPU}"
        echo "=== ${MODEL} | GPU ${GPU} | tp=${TP_SIZE} ===" > "${LOG}"
        echo "Start: $(date)" >> "${LOG}"

        python run_evaluation.py \
            --model_path "${MODEL_PATH}" \
            --mode "${MODE}" \
            --k "${K}" \
            --benchmarks aime24 aime25 math500 \
            --tensor_parallel_size "${TP_SIZE}" \
            --gpu_memory_utilization 0.92 \
            --timestamp "${TS}" \
            >> "${LOG}" 2>&1

        rc=$?
        echo "End: $(date) | rc=${rc}" >> "${LOG}"
        echo "[Done] ${MODEL} rc=${rc}" | tee -a "${LOG_DIR}/progress.log"
        exit $rc
    ) &
    PIDS+=($!)
done

echo "All jobs launched, waiting..."

failed=0
for pid in "${PIDS[@]}"; do
    wait "$pid" || failed=1
done

echo ""
echo "=================================================="
if [ "$failed" -ne 0 ]; then
    echo "Some evaluations FAILED. Check logs in ${LOG_DIR}"
else
    echo "All evaluations completed successfully!"
fi
echo "=================================================="

echo ""
echo "=== Quick Results ==="
for model in "${MODELS[@]}"; do
    echo "--- ${model} ---"
    grep -E "pass@1|accuracy_any|Pass@1" "${LOG_DIR}/${model}.log" 2>/dev/null | tail -10
done

#!/bin/bash
set -euo pipefail
set -x

python -m pip install "omegaconf==2.3.0" "tensordict==0.10.0" "torchdata==0.11.0" "codetiming==1.4.0"
pip install -U hydra-core
pip uninstall -y flash-attn
pip install -v --no-build-isolation flash-attn

export VLLM_DISABLE_CUSTOM_ALL_REDUCE=1
export VLLM_USE_SYMM_MEM=0


cd S2L-PO
# ========================================
OFFLINE_ROLLOUT_DATA="/data/4B_rollout.jsonl"
ONLINE_TRAIN_DATA="/data/train_dedup.parquet"
VAL_DATA="/data/test.parquet"

INIT_MODEL_PATH="/Qwen3-8B-Base"
CKPT_DIR="/checkpoints/qwen3_4b_8b_16_off2on_8_s"

# ========================================
N=${1:-8}             
TOTAL_LOGICAL_STEPS=16  
BATCH_SIZE=1024
TOTAL_ROLLOUTS=16       
TOTAL_OFFLINE_SAMPLES=16 

# ========================================
COMMON_ARGS=(
  algorithm.adv_estimator=grpo

  data.train_batch_size=${BATCH_SIZE}
  data.max_prompt_length=512
  data.max_response_length=4096
  data.filter_overlong_prompts=True
  data.truncation='error'

  actor_rollout_ref.model.use_remove_padding=True
  actor_rollout_ref.model.enable_gradient_checkpointing=True
  actor_rollout_ref.actor.optim.lr=1e-6
  actor_rollout_ref.actor.ppo_mini_batch_size=32
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4

  actor_rollout_ref.actor.use_kl_loss=True
  actor_rollout_ref.actor.kl_loss_coef=0.001
  actor_rollout_ref.actor.kl_loss_type=low_var_kl
  actor_rollout_ref.actor.entropy_coeff=0
  actor_rollout_ref.actor.fsdp_config.param_offload=False
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False

  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4
  actor_rollout_ref.rollout.tensor_model_parallel_size=1
  actor_rollout_ref.rollout.name=vllm
  actor_rollout_ref.rollout.gpu_memory_utilization=0.5

  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4
  actor_rollout_ref.ref.fsdp_config.param_offload=False

  algorithm.use_kl_in_reward=False

  trainer.critic_warmup=0
  trainer.logger='["console"]'
  trainer.project_name='default'
  trainer.experiment_name=''
  trainer.n_gpus_per_node=8
  trainer.nnodes=1
  trainer.save_freq=1
  trainer.test_freq=100
  trainer.default_local_dir=${CKPT_DIR}
  trainer.resume_mode=auto
  trainer.total_epochs=999999
)

# ========================================
get_global_step () {
  local f="${CKPT_DIR}/latest_checkpointed_iteration.txt"
  if [[ -f "${f}" ]]; then
    cat "${f}"
  else
    echo "0"
  fi
}


get_logical_step_done () {
  local f="${CKPT_DIR}/logical_step_progress.txt"
  if [[ -f "${f}" ]]; then
    cat "${f}"
  else
    echo "0"
  fi
}

save_logical_step_done () {
  local logical_step="$1"
  echo "${logical_step}" > "${CKPT_DIR}/logical_step_progress.txt"
}

run_offline () {
  local end_step="$1"
  local n_samples="$2"
  echo "  → Running OFFLINE to step ${end_step} with offline_samples_per_prompt=${n_samples}"
  python3 -m verl.trainer.main_ppo \
    "${COMMON_ARGS[@]}" \
    trainer.total_training_steps="${end_step}" \
    actor_rollout_ref.model.path="${INIT_MODEL_PATH}" \
    data.train_files="${OFFLINE_ROLLOUT_DATA}" \
    data.val_files="${VAL_DATA}" \
    +data.use_offline_rollout=True \
    +data.offline_samples_per_prompt="${n_samples}" \
    +data.offline_load_rollout_log_probs=False \
    actor_rollout_ref.rollout.skip_rollout=True
}

run_online () {
  local end_step="$1"
  local n_rollouts="$2"
  echo "  → Running ONLINE to step ${end_step} with n=${n_rollouts} rollouts"
  python3 -m verl.trainer.main_ppo \
    "${COMMON_ARGS[@]}" \
    trainer.total_training_steps="${end_step}" \
    actor_rollout_ref.model.path="${INIT_MODEL_PATH}" \
    data.train_files="${ONLINE_TRAIN_DATA}" \
    data.val_files="${VAL_DATA}" \
    actor_rollout_ref.rollout.n="${n_rollouts}"
}

# ========================================
initial_global_step=$(get_global_step)
initial_logical_step=$(get_logical_step_done)

echo "=========================================="
echo "Progressive Training: N=${N} logical steps, Total=${TOTAL_LOGICAL_STEPS} logical steps"
echo "Total rollouts per step (online): ${TOTAL_ROLLOUTS}"
echo "Total offline samples per prompt: ${TOTAL_OFFLINE_SAMPLES}"
echo "----------------------------------------"
echo "Resume from: global_step=${initial_global_step}, logical_step=${initial_logical_step}"
echo "=========================================="

# ========================================
logical_step_done="${initial_logical_step}"

for t in $(seq 1 "${N}"); do
  if [[ "${t}" -le "${logical_step_done}" ]]; then
    echo "[Logical Step ${t}/${N}] Already completed, skipping..."
    continue
  fi

  curr=$(get_global_step)

  if [[ "${N}" -le 1 ]]; then
    offline_ratio=1.0
  else
    offline_ratio=$(python3 -c "print(1 - ($t-1)/($N-1))")
  fi

  offline_samples=$(python3 -c "import math; print(int(math.ceil(${offline_ratio} * ${TOTAL_OFFLINE_SAMPLES})))")
  online_rollouts=$(python3 -c "import math; print(int(math.ceil((1-${offline_ratio}) * ${TOTAL_ROLLOUTS})))")

  echo ""
  echo "[Logical Step ${t}/${N}] Offline ratio: ${offline_ratio}"
  echo "  Current global_step: ${curr}"
  echo "  Offline samples: ${offline_samples}/${TOTAL_OFFLINE_SAMPLES}, Online rollouts: ${online_rollouts}/${TOTAL_ROLLOUTS}"

  if [[ "${offline_samples}" -gt 0 ]]; then
    run_offline $((curr + 1)) "${offline_samples}"
    curr=$((curr + 1))
  fi

  if [[ "${online_rollouts}" -gt 0 ]]; then
    run_online $((curr + 1)) "${online_rollouts}"
    curr=$((curr + 1))
  fi

  logical_step_done=$((logical_step_done + 1))
  save_logical_step_done "${logical_step_done}"

  actual=$(get_global_step)
  echo "  After logical step ${t}: global_step=${actual}, logical_step_done=${logical_step_done}"
done

# ========================================
remaining_logical=$((TOTAL_LOGICAL_STEPS - logical_step_done))

if [[ "${remaining_logical}" -gt 0 ]]; then
  echo ""
  echo "=========================================="
  echo "Remaining ${remaining_logical} logical steps: 100% ONLINE"
  echo "=========================================="

  for i in $(seq 1 "${remaining_logical}"); do
    curr=$(get_global_step)
    echo "[Remaining step ${i}/${remaining_logical}] Running ONLINE"
    run_online $((curr + 1)) "${TOTAL_ROLLOUTS}"

    logical_step_done=$((logical_step_done + 1))
    save_logical_step_done "${logical_step_done}"
  done
fi

final_step=$(get_global_step)
echo ""
echo "=========================================="
echo "✓ Training completed!"
echo "Total physical steps: ${final_step}"
echo "Total logical steps: ${TOTAL_LOGICAL_STEPS}"
echo "Final checkpoint: ${CKPT_DIR}"
echo "=========================================="
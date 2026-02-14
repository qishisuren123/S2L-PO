#!/bin/bash
set -euo pipefail
set -x
export HTTP_PROXY=http://sys-proxy-rd-relay.byted.org:8118
export http_proxy=http://sys-proxy-rd-relay.byted.org:8118
export https_proxy=http://sys-proxy-rd-relay.byted.org:8118
export no_proxy="localhost,.byted.org,byted.org,.bytedance.net,bytedance.net,.byteintl.net,.tiktok-row.net,.tiktok-row.org,127.0.0.1,127.0.0.0/8,169.254.0.0/16,100.64.0.0/10,172.16.0.0/12,192.168.0.0/16,10.0.0.0/8,::1,fe80::/10,fd00::/8"
python -m pip install "omegaconf==2.3.0" "tensordict==0.10.0" "torchdata==0.11.0" "codetiming==1.4.0"
pip install -U hydra-core
# pip3 install --no-build-isolation axolotl[flash-attn,deepspeed]
# # pip uninstall -y flash-attn flash_attn
# export TRANSFORMERS_FLASH_ATTENTION=0
# pip install flash-attn==2.8.3
# python3 -m pip install -r <(grep -v '^merlin_kernel @ file:///opt/tiger/mlx_deploy/python_kernel/1.0.0.474$' /mnt/bn/protenix-3/xuyiran/yr/requirements.txt)
pip uninstall -y flash-attn
pip install -v --no-build-isolation flash-attn
export VLLM_DISABLE_CUSTOM_ALL_REDUCE=1
export VLLM_USE_SYMM_MEM=0
cd /mnt/bn/protenix-3/xuyiran/yr/e-GRPO
# ==================== 路径配置 ====================
OFFLINE_ROLLOUT_DATA="/mnt/bn/protenix-3/xuyiran/yr/datasets/aime24/4B_rollout_noprob.jsonl"
# OFFLINE_ROLLOUT_DATA="/mnt/bn/protenix-3/xuyiran/yr/e-GRPO/small_model_rollouts_test/small_model_rollouts.jsonl"
ONLINE_TRAIN_DATA="/mnt/bn/protenix-3/xuyiran/yr/datasets/dapo17k/train_dedup.parquet"
VAL_DATA="/mnt/bn/protenix-3/xuyiran/yr/datasets/gsm8k/test.parquet"
INIT_MODEL_PATH="/mnt/bn/protenix-3/xuyiran/yr/qwen_weight/Qwen3-14B-Base"
CKPT_DIR="/mnt/bn/protenix-3/xuyiran/yr/e-GRPO/checkpoints/qwen3_4b_14b_8_off2on_4_s"
# ==================== 关键参数 ====================
N=${1:-4}              # 渐进切换的逻辑步数
TOTAL_LOGICAL_STEPS=16  # 数据集的逻辑步数（16k/1k）
BATCH_SIZE=1024
TOTAL_ROLLOUTS=8        # 在线rollout数
TOTAL_OFFLINE_SAMPLES=8  # 离线数据每个prompt的总样本数
# ==================== 固定训练超参 ====================
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
  actor_rollout_ref.actor.ppo_mini_batch_size=16
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2
  actor_rollout_ref.actor.use_kl_loss=True
  actor_rollout_ref.actor.kl_loss_coef=0.001
  actor_rollout_ref.actor.kl_loss_type=low_var_kl
  actor_rollout_ref.actor.entropy_coeff=0
  actor_rollout_ref.actor.fsdp_config.param_offload=False
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2
  actor_rollout_ref.rollout.tensor_model_parallel_size=1
  actor_rollout_ref.rollout.name=vllm
  actor_rollout_ref.rollout.gpu_memory_utilization=0.5
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2
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
# ==================== 工具函数 ====================
get_global_step () {
  local f="${CKPT_DIR}/latest_checkpointed_iteration.txt"
  if [[ -f "${f}" ]]; then
    cat "${f}"
  else
    echo "0"
  fi
}
# 新增：获取已完成的逻辑步数
get_logical_step_done () {
  local f="${CKPT_DIR}/logical_step_progress.txt"
  if [[ -f "${f}" ]]; then
    cat "${f}"
  else
    echo "0"
  fi
}
# 新增：保存逻辑步进度
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
# ==================== 断点续训检测 ====================
initial_global_step=$(get_global_step)
initial_logical_step=$(get_logical_step_done)
echo "=========================================="
echo "Progressive Training: N=${N} logical steps, Total=${TOTAL_LOGICAL_STEPS} logical steps"
echo "Total rollouts per step (online): ${TOTAL_ROLLOUTS}"
echo "Total offline samples per prompt: ${TOTAL_OFFLINE_SAMPLES}"
echo "----------------------------------------"
echo "Resume from: global_step=${initial_global_step}, logical_step=${initial_logical_step}"
echo "=========================================="
# ==================== 渐进调度（前N个逻辑步） ====================
logical_step_done="${initial_logical_step}"
for t in $(seq 1 "${N}"); do
  # 跳过已完成的逻辑步
  if [[ "${t}" -le "${logical_step_done}" ]]; then
    echo "[Logical Step ${t}/${N}] Already completed, skipping..."
    continue
  fi
  curr=$(get_global_step)
  # 计算离线比例
  if [[ "${N}" -le 1 ]]; then
    offline_ratio=1.0
  else
    offline_ratio=$(python3 -c "print(1 - ($t-1)/($N-1))")
  fi
  # 计算离线样本数和在线rollout数
  offline_samples=$(python3 -c "import math; print(int(math.ceil(${offline_ratio} * ${TOTAL_OFFLINE_SAMPLES})))")
  online_rollouts=$(python3 -c "import math; print(int(math.ceil((1-${offline_ratio}) * ${TOTAL_ROLLOUTS})))")
  echo ""
  echo "[Logical Step ${t}/${N}] Offline ratio: ${offline_ratio}"
  echo "  Current global_step: ${curr}"
  echo "  Offline samples: ${offline_samples}/${TOTAL_OFFLINE_SAMPLES}, Online rollouts: ${online_rollouts}/${TOTAL_ROLLOUTS}"
  # 先跑离线部分
  if [[ "${offline_samples}" -gt 0 ]]; then
    run_offline $((curr + 1)) "${offline_samples}"
    curr=$((curr + 1))
  fi
  # 再跑在线部分
  if [[ "${online_rollouts}" -gt 0 ]]; then
    run_online $((curr + 1)) "${online_rollouts}"
    curr=$((curr + 1))
  fi
  # 更新逻辑步进度
  logical_step_done=$((logical_step_done + 1))
  save_logical_step_done "${logical_step_done}"
  actual=$(get_global_step)
  echo "  After logical step ${t}: global_step=${actual}, logical_step_done=${logical_step_done}"
done
# ==================== 剩余逻辑步全部在线 ====================
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
    # 更新逻辑步进度
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
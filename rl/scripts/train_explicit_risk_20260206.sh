#!/bin/bash
# 训练脚本: Qwen3-4B-Instruct-Explicit-Risk
# 描述: 显式风险阈值训练 (Section 3.2.1 / Figure 1)
# 说明: 训练时随机采样 t~U(0,1)，并在prompt中显式给出

set -e

# 激活conda环境（兼容不同安装路径）
if [ -f "/data1/conda/etc/profile.d/conda.sh" ]; then
    source /data1/conda/etc/profile.d/conda.sh
elif [ -f "/workspace/miniconda3/etc/profile.d/conda.sh" ]; then
    source /workspace/miniconda3/etc/profile.d/conda.sh
elif command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
else
    echo "未找到 conda，请先安装或设置正确路径。"
    exit 1
fi
conda activate truthrl-verl-npu

# 激活CANN环境（如存在）
if [ -f "/usr/local/Ascend/ascend-toolkit/set_env.sh" ]; then
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi
if [ -f "/usr/local/Ascend/nnal/atb/set_env.sh" ]; then
    source /usr/local/Ascend/nnal/atb/set_env.sh
fi

# 设置HuggingFace镜像
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_CACHE=/data2/cache/huggingface
export HF_HUB_OFFLINE=0
export TRANSFORMERS_OFFLINE=0
export HF_DATASETS_OFFLINE=0

# NPU & vLLM 相关环境变量
export VLLM_DEVICE=npu
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_ASCEND_ENABLE_NZ=0
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export WANDB_MODE=online
export WANDB_API_KEY="d0e4ffd3de59b61bc1bd9b069a15a76c4b3d9927"
export HCCL_PORT_RANGE=40000-40999

# 数据路径（当前实验：NQ/Hotpot 小样本）
TRAIN_FILES="/root/code/Fact_Reasoning/GT_GRPO/data/nq_hotpot_searchr1/train_8000.parquet"
VAL_FILES="/root/code/Fact_Reasoning/GT_GRPO/data/nq_hotpot_searchr1/test_1000.parquet"

# 自定义奖励函数
REWARD_FUNCTION_PATH="/root/code/SeedMitigating/rl/verl_custom_reward.py"

# 自定义数据集（注入prompt模板）
CUSTOM_DATASET_PATH="/root/code/SeedMitigating/rl/behavioral_dataset.py"

PROJECT_NAME="behavioral_calibration"
EXPERIMENT_NAME="explicit_risk"

OUTPUT_DIR="/root/code/SeedMitigating/output/checkpoints"
OUTPUT_DIR="${OUTPUT_DIR}/Qwen3-4B-Instruct-Explicit-Risk-$(date +%Y%m%d_%H%M%S)"
rm -f "$OUTPUT_DIR/TRAINING_DONE"
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "训练: Qwen3-4B-Instruct-Explicit-Risk"
echo "=========================================="
echo "策略: 显式风险阈值 (t~U[0,1])"
# echo "奖励: +1 (正确), 0 (IDK), -t/(1-t) (错误)"
echo "奖励: +1 (正确), -1 (IDK), -(1+t)/(1-t) (错误)"
echo "算法: GRPO"
echo "输出: $OUTPUT_DIR"

echo ""

# 运行训练
cd "/root/code/SeedMitigating/verl"

python3 -m verl.trainer.main_ppo \
    --config-path=config \
    --config-name=ppo_trainer \
    algorithm.adv_estimator=grpo \
    algorithm.gamma=1.0 \
    algorithm.lam=1.0 \
    algorithm.norm_adv_by_std_in_grpo=True \
    data.train_files="['$TRAIN_FILES']" \
    data.val_files="['$VAL_FILES']" \
    data.train_batch_size=256 \
    data.val_batch_size=256 \
    data.train_max_samples=5000 \
    data.max_prompt_length=512 \
    data.max_response_length=1536 \
    data.reward_fn_key=data_source \
    data.custom_cls.path="$CUSTOM_DATASET_PATH" \
    data.custom_cls.name=BehavioralCalibrationDataset \
    +data.strategy=explicit_risk \
    actor_rollout_ref.model.path=Qwen/Qwen3-4B-Instruct-2507 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.7 \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=16 \
    critic.model.path=Qwen/Qwen3-4B-Instruct-2507 \
    critic.optim.lr=1e-5 \
    critic.optim.lr_warmup_steps=10 \
    critic.optim.weight_decay=0.01 \
    critic.ppo_micro_batch_size_per_gpu=4 \
    critic.model.external_lib=trl \
    algorithm.use_kl_in_reward=False \
    custom_reward_function.path="$REWARD_FUNCTION_PATH" \
    custom_reward_function.name=compute_score_explicit_risk \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="$PROJECT_NAME" \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=10 \
    trainer.total_epochs=30 \
    trainer.critic_warmup=20 \
    trainer.resume_mode=disable \
    trainer.max_actor_ckpt_to_keep=2 \
    trainer.max_critic_ckpt_to_keep=2 \
    trainer.device=npu \
    trainer.default_local_dir="$OUTPUT_DIR" \
    reward_model.enable=False \
    "$@"

echo ""
echo "=========================================="
touch "$OUTPUT_DIR/TRAINING_DONE"
echo "✓ 训练完成: Explicit Risk"
echo "模型保存在: $OUTPUT_DIR"
echo "=========================================="

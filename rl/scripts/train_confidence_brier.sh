#!/bin/bash
# 训练脚本: Qwen3-4B-Instruct-Confidence-Brier
# 描述: Verbalized Confidence + Brier Score (均匀分布风险阈值)
# 对应论文: Section 3.2.2 - Uniform Distribution

set -e

# 激活conda环境
source /data1/conda/etc/profile.d/conda.sh
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
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# NPU & vLLM 相关环境变量
export VLLM_DEVICE=npu
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_ASCEND_ENABLE_NZ=0
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export WANDB_MODE=offline

# 数据路径
TRAIN_FILES="/root/SeedMitigating/data/dapo_math_train.parquet"
VAL_FILES="/root/SeedMitigating/data/dapo_math_val.parquet"

# 自定义奖励函数
REWARD_FUNCTION_PATH="/root/SeedMitigating/rl/verl_custom_reward.py"

# 自定义数据集（注入prompt模板）
CUSTOM_DATASET_PATH="/root/SeedMitigating/rl/behavioral_dataset.py"

# 输出目录
OUTPUT_DIR="/data2/SeedMitigating-output/paper_reproduction/checkpoints/confidence_brier"
rm -f "$OUTPUT_DIR/TRAINING_DONE"
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "训练: Qwen3-4B-Instruct-Confidence-Brier"
echo "=========================================="
echo "策略: Verbalized Confidence + Brier Score"
echo "奖励: R = 2p·valid(y) - p²"
echo "风险阈值分布: 均匀分布"
echo "输出: $OUTPUT_DIR"

echo ""

# 运行训练
cd "/root/SeedMitigating/verl"

python3 -m verl.trainer.main_ppo \
    --config-path=config \
    --config-name=ppo_trainer \
    algorithm.adv_estimator=gae \
    algorithm.gamma=1.0 \
    algorithm.lam=1.0 \
    data.train_files="['$TRAIN_FILES']" \
    data.val_files="['$VAL_FILES']" \
    data.train_batch_size=512 \
    data.val_batch_size=512 \
    data.max_prompt_length=512 \
    data.max_response_length=20480 \
    data.reward_fn_key=data_source \
    data.custom_cls.path="$CUSTOM_DATASET_PATH" \
    data.custom_cls.name=BehavioralCalibrationDataset \
    +data.strategy=verbalized_brier \
    actor_rollout_ref.model.path=Qwen/Qwen3-4B-Instruct-2507 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
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
    algorithm.kl_ctrl.kl_coef=0.0 \
    custom_reward_function.path="$REWARD_FUNCTION_PATH" \
    custom_reward_function.name=compute_score_verbalized_brier \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=behavioral_calibration_paper \
    trainer.experiment_name=confidence_brier \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=5 \
    trainer.test_freq=1 \
    trainer.total_epochs=30 \
    trainer.critic_warmup=20 \
    trainer.device=npu \
    trainer.default_local_dir="$OUTPUT_DIR" \
    reward_model.enable=False \
    reward_model.reward_manager=dapo \
    +reward_model.reward_kwargs.max_resp_len=20480 \
    +reward_model.reward_kwargs.overlong_buffer_cfg.enable=True \
    +reward_model.reward_kwargs.overlong_buffer_cfg.len=4096 \
    +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0 \
    +reward_model.reward_kwargs.overlong_buffer_cfg.log=False \
    "$@"

echo ""
echo "=========================================="
touch "$OUTPUT_DIR/TRAINING_DONE"
echo "✓ 训练完成: Confidence-Brier"
echo "模型保存在: $OUTPUT_DIR"
echo "=========================================="

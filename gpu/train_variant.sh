#!/bin/bash

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: bash gpu/train_variant.sh <variant> [hydra overrides...]"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"

VARIANT="$1"
shift || true

TRAIN_FILES="${SEEDMIT_TRAIN_FILES:-$SEEDMIT_PROJECT_ROOT/data/dapo_math_train.parquet}"
VAL_FILES="${SEEDMIT_VAL_FILES:-$SEEDMIT_PROJECT_ROOT/data/dapo_math_val.parquet}"
REWARD_FUNCTION_PATH="$SEEDMIT_PROJECT_ROOT/rl/verl_custom_reward.py"
CUSTOM_DATASET_PATH="$SEEDMIT_PROJECT_ROOT/rl/behavioral_dataset.py"
OUTPUT_DIR="${SEEDMIT_OUTPUT_BASE}/checkpoints/${VARIANT}"

mkdir -p "$OUTPUT_DIR"
rm -f "$OUTPUT_DIR/TRAINING_DONE"

DATA_STRATEGY=""
REWARD_NAME=""
ADV_ESTIMATOR="gae"
REWARD_EXTRA=()

case "$VARIANT" in
    baseline_ppo)
        DATA_STRATEGY="baseline"
        REWARD_NAME="compute_score_baseline"
        ;;
    confidence_brier)
        DATA_STRATEGY="verbalized_brier"
        REWARD_NAME="compute_score_verbalized_brier"
        ;;
    confidence_ce)
        DATA_STRATEGY="verbalized_ce"
        REWARD_NAME="compute_score_verbalized_ce"
        ADV_ESTIMATOR="grpo"
        REWARD_EXTRA=(+custom_reward_function.reward_kwargs.epsilon=0.01 algorithm.norm_adv_by_std_in_grpo=True)
        ;;
    ppo_value)
        DATA_STRATEGY="ppo_value"
        REWARD_NAME="compute_score_baseline"
        ;;
    confidence_prod)
        DATA_STRATEGY="claim_product"
        REWARD_NAME="compute_score_claim_product"
        ;;
    confidence_min)
        DATA_STRATEGY="claim_minimum"
        REWARD_NAME="compute_score_claim_minimum"
        ;;
    explicit_risk)
        DATA_STRATEGY="explicit_risk"
        REWARD_NAME="compute_score_explicit_risk"
        ;;
    *)
        echo "Unknown variant: $VARIANT"
        exit 1
        ;;
esac

echo "=========================================="
echo "Training variant: $VARIANT"
echo "Project root    : $SEEDMIT_PROJECT_ROOT"
echo "Output dir      : $OUTPUT_DIR"
echo "Visible GPUs    : $SEEDMIT_NUM_GPUS"
echo "=========================================="

cd "$SEEDMIT_PROJECT_ROOT/verl"

"$SEEDMIT_PYTHON" -m verl.trainer.main_ppo \
    --config-path=config \
    --config-name=ppo_trainer \
    algorithm.adv_estimator="$ADV_ESTIMATOR" \
    algorithm.gamma=1.0 \
    algorithm.lam=1.0 \
    data.train_files="['$TRAIN_FILES']" \
    data.val_files="['$VAL_FILES']" \
    data.train_batch_size="$SEEDMIT_TRAIN_BATCH_SIZE" \
    data.val_batch_size="$SEEDMIT_VAL_BATCH_SIZE" \
    data.max_prompt_length="$SEEDMIT_MAX_PROMPT_LENGTH" \
    data.max_response_length="$SEEDMIT_MAX_RESPONSE_LENGTH" \
    data.reward_fn_key=data_source \
    data.custom_cls.path="$CUSTOM_DATASET_PATH" \
    data.custom_cls.name=BehavioralCalibrationDataset \
    +data.strategy="$DATA_STRATEGY" \
    actor_rollout_ref.model.path="$SEEDMIT_BASE_MODEL" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size="$SEEDMIT_PPO_MINI_BATCH_SIZE" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$SEEDMIT_PPO_MICRO_BATCH_SIZE" \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n="$SEEDMIT_ROLLOUT_N" \
    actor_rollout_ref.rollout.gpu_memory_utilization="$SEEDMIT_ROLLOUT_GPU_MEMORY_UTIL" \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.7 \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size="$SEEDMIT_LOGPROB_MICRO_BATCH" \
    critic.model.path="$SEEDMIT_BASE_MODEL" \
    critic.optim.lr=1e-5 \
    critic.optim.lr_warmup_steps=10 \
    critic.optim.weight_decay=0.01 \
    critic.ppo_micro_batch_size_per_gpu="$SEEDMIT_CRITIC_MICRO_BATCH_SIZE" \
    critic.model.external_lib=trl \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.0 \
    custom_reward_function.path="$REWARD_FUNCTION_PATH" \
    custom_reward_function.name="$REWARD_NAME" \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=behavioral_calibration_gpu \
    trainer.experiment_name="$VARIANT" \
    trainer.n_gpus_per_node="$SEEDMIT_NUM_GPUS" \
    trainer.nnodes=1 \
    trainer.save_freq="$SEEDMIT_SAVE_FREQ" \
    trainer.test_freq="$SEEDMIT_TEST_FREQ" \
    trainer.total_epochs="$SEEDMIT_TRAIN_EPOCHS" \
    trainer.critic_warmup="$SEEDMIT_CRITIC_WARMUP" \
    trainer.device=cuda \
    trainer.default_local_dir="$OUTPUT_DIR" \
    reward_model.enable=False \
    reward_model.reward_manager=dapo \
    +reward_model.reward_kwargs.max_resp_len="$SEEDMIT_MAX_RESPONSE_LENGTH" \
    +reward_model.reward_kwargs.overlong_buffer_cfg.enable=True \
    +reward_model.reward_kwargs.overlong_buffer_cfg.len=4096 \
    +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0 \
    +reward_model.reward_kwargs.overlong_buffer_cfg.log=False \
    "${REWARD_EXTRA[@]}" \
    "$@"

touch "$OUTPUT_DIR/TRAINING_DONE"
echo "Training finished: $VARIANT"

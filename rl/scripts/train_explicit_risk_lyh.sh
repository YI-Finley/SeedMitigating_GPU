#!/bin/bash
ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
# 训练脚本: Qwen3-4B-Instruct-Explicit-Risk
# 描述: 显式风险阈值训练 (Section 3.2.1 / Figure 1)
# 说明: 训练时每条样本随机采样 t~U(0,1)，并在 prompt 中显式给出（不支持固定 t）
# 默认: HotpotQA(SearchR1) 采样 16 条、8 NPU、1 epoch
# 用法示例:
#   bash rl/scripts/train_explicit_risk.sh
#   SAMPLE_SIZE=100 bash rl/scripts/train_explicit_risk.sh
#   USE_SAMPLE=0 TRAIN_FILES=/path/train.parquet VAL_FILES=/path/val.parquet bash rl/scripts/train_explicit_risk.sh

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

# 模型路径（优先使用本地 snapshot，离线环境避免走远端）
MODEL_ID="Qwen/Qwen3-4B-Instruct-2507"
MODEL_SNAPSHOT="/data2/cache/huggingface/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554"
if [ -d "$MODEL_SNAPSHOT" ]; then
    MODEL_PATH="$MODEL_SNAPSHOT"
else
    MODEL_PATH="$MODEL_ID"
fi

# NPU & vLLM 相关环境变量
export VLLM_DEVICE=npu
# vLLM-Ascend 在 dev 版本上需要显式指定版本，避免补丁分支不匹配
export VLLM_VERSION="${VLLM_VERSION:-0.11.0}"
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_ASCEND_ENABLE_NZ=0
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export WANDB_MODE=offline
# 输出更多日志便于定位退出原因
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"
export RAY_DEDUP_LOGS="${RAY_DEDUP_LOGS:-0}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
# 提升通信稳定性（AIV 扩展模式）
export HCCL_OP_EXPANSION_MODE=AIV

# 数据与采样配置（默认 HotpotQA SearchR1）
SAMPLE_SIZE="${SAMPLE_SIZE:-16}"
USE_SAMPLE="${USE_SAMPLE:-1}"
DATA_DIR="$ROOT_DIR/external/GT_GRPO/data/nq_hotpot_searchr1"
ALT_DATA_DIR="/root/LYH/SeedMitigating/external/GT_GRPO/data/nq_hotpot_searchr1"
if [ ! -f "$DATA_DIR/train.parquet" ] && [ -f "$ALT_DATA_DIR/train.parquet" ]; then
    echo "[WARN] 默认数据不存在，改用 $ALT_DATA_DIR"
    DATA_DIR="$ALT_DATA_DIR"
fi
SRC_TRAIN="${SRC_TRAIN:-$DATA_DIR/train.parquet}"
SRC_VAL="${SRC_VAL:-$DATA_DIR/test.parquet}"
TRAIN_FILES="${TRAIN_FILES:-}"
VAL_FILES="${VAL_FILES:-}"

# Debug：默认打开（可通过环境变量关闭）
EXPLICIT_RISK_DEBUG="${EXPLICIT_RISK_DEBUG:-1}"
EXPLICIT_RISK_DEBUG_PROMPTS="${EXPLICIT_RISK_DEBUG_PROMPTS:-1}"
EXPLICIT_RISK_DEBUG_REWARD="${EXPLICIT_RISK_DEBUG_REWARD:-1}"
export EXPLICIT_RISK_DEBUG EXPLICIT_RISK_DEBUG_PROMPTS EXPLICIT_RISK_DEBUG_REWARD

# 日志重定向
LOG_DIR="$ROOT_DIR/output/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/explicit_risk_hotpotqa_sample${SAMPLE_SIZE}_randt_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "[INFO] 日志输出: $LOG_FILE"

# 采样逻辑（默认开启）
if [ -n "$TRAIN_FILES" ]; then
    SRC_TRAIN="$TRAIN_FILES"
fi
if [ -n "$VAL_FILES" ]; then
    SRC_VAL="$VAL_FILES"
fi

if [ "$USE_SAMPLE" != "0" ]; then
    TMP_DIR="/tmp/seedmitigating_hotpotqa_sample${SAMPLE_SIZE}"
    mkdir -p "$TMP_DIR"
    TRAIN_FILES="$TMP_DIR/train_sample.parquet"
    VAL_FILES="$TMP_DIR/val_sample.parquet"
    export SRC_TRAIN SRC_VAL TRAIN_FILES VAL_FILES SAMPLE_SIZE
    python3 - <<'PY'
import os
import pyarrow as pa
import pyarrow.parquet as pq

src_train = os.environ.get("SRC_TRAIN")
src_val = os.environ.get("SRC_VAL")
out_train = os.environ.get("TRAIN_FILES")
out_val = os.environ.get("VAL_FILES")
sample_size = int(os.environ.get("SAMPLE_SIZE", "16"))

def build_sample(src_path, out_path, n):
    if not os.path.exists(src_path):
        raise FileNotFoundError(src_path)
    table = pq.read_table(src_path)
    n = max(1, min(n, table.num_rows))
    table = table.slice(0, n)
    data = table.to_pydict()
    golden = data.get("golden_answers") or []
    data_source = data.get("data_source")
    if data_source is None:
        data["data_source"] = ["nq_hotpotqa_searchr1"] * len(golden)
    reward_model = []
    for ga in golden:
        reward_model.append({"ground_truth": {"target": ga}, "style": "rule"})
    data["reward_model"] = reward_model
    out_table = pa.table(data)
    pq.write_table(out_table, out_path)

build_sample(src_train, out_train, sample_size)
build_sample(src_val, out_val, sample_size)
print(f"[INFO] Wrote {out_train} and {out_val}")
PY
else
    if [ -z "$TRAIN_FILES" ]; then
        TRAIN_FILES="$SRC_TRAIN"
    fi
    if [ -z "$VAL_FILES" ]; then
        VAL_FILES="$SRC_VAL"
    fi
fi

# 自定义奖励函数
REWARD_FUNCTION_PATH="$ROOT_DIR/rl/verl_custom_reward.py"

# 自定义数据集（注入prompt模板）
CUSTOM_DATASET_PATH="$ROOT_DIR/rl/behavioral_dataset.py"

# 输出目录
if [ -z "${OUTPUT_DIR:-}" ]; then
    if [ "$USE_SAMPLE" != "0" ]; then
        OUTPUT_DIR="/data2/SeedMitigating-output/paper_reproduction/checkpoints/explicit_risk_ws8_sample${SAMPLE_SIZE}_n2"
    else
        OUTPUT_DIR="/data2/SeedMitigating-output/paper_reproduction/checkpoints/explicit_risk"
    fi
fi
rm -f "$OUTPUT_DIR/TRAINING_DONE"
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "训练: Qwen3-4B-Instruct-Explicit-Risk"
echo "=========================================="
echo "策略: 显式风险阈值 (t~U[0,1])"
echo "奖励: +1 (正确), 0 (IDK), -t/(1-t) (错误)"
echo "算法: GRPO"
echo "输出: $OUTPUT_DIR"
echo "数据: $TRAIN_FILES"

echo ""

# 运行训练
cd "$ROOT_DIR/verl"

EXTRA_DATA_ARGS=()
if [ "$EXPLICIT_RISK_DEBUG" = "1" ]; then
    EXTRA_DATA_ARGS+=(+data.debug_explicit_risk=true)
fi
if [ "$EXPLICIT_RISK_DEBUG_PROMPTS" = "1" ]; then
    EXTRA_DATA_ARGS+=(+data.debug_explicit_risk_prompts=true)
fi

python3 -m verl.trainer.main_ppo \
    --config-path=config \
    --config-name=ppo_trainer \
    algorithm.adv_estimator=grpo \
    algorithm.gamma=1.0 \
    algorithm.lam=1.0 \
    algorithm.norm_adv_by_std_in_grpo=True \
    data.train_files="['$TRAIN_FILES']" \
    data.val_files="['$VAL_FILES']" \
    data.train_batch_size=8 \
    data.val_batch_size=8 \
    data.max_prompt_length=512 \
    data.max_response_length=512 \
    data.dataloader_num_workers=0 \
    data.reward_fn_key=data_source \
    data.custom_cls.path="$CUSTOM_DATASET_PATH" \
    data.custom_cls.name=BehavioralCalibrationDataset \
    +data.strategy=explicit_risk \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.7 \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=16 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.response_length=512 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enable_prefix_caching=False \
    actor_rollout_ref.rollout.max_model_len=2048 \
    actor_rollout_ref.rollout.max_num_batched_tokens=2048 \
    actor_rollout_ref.rollout.max_num_seqs=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    critic.model.path="$MODEL_PATH" \
    critic.optim.lr=1e-5 \
    critic.optim.lr_warmup_steps=10 \
    critic.optim.weight_decay=0.01 \
    critic.ppo_micro_batch_size_per_gpu=1 \
    critic.model.external_lib=trl \
    +critic.use_torch_compile=False \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.0 \
    custom_reward_function.path="$REWARD_FUNCTION_PATH" \
    custom_reward_function.name=compute_score_explicit_risk \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=behavioral_calibration_paper \
    trainer.experiment_name=explicit_risk \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=1 \
    trainer.test_freq=1 \
    trainer.total_epochs=1 \
    trainer.critic_warmup=20 \
    trainer.device=npu \
    trainer.default_local_dir="$OUTPUT_DIR" \
    trainer.resume_mode=disable \
    reward_model.enable=False \
    reward_model.reward_manager=naive \
    +actor_rollout_ref.model.override_config.attn_implementation=eager \
    +critic.model.override_config.attn_implementation=eager \
    +actor_rollout_ref.ref.model.override_config.attn_implementation=eager \
    "${EXTRA_DATA_ARGS[@]}" \
    "$@"

echo ""
echo "=========================================="
touch "$OUTPUT_DIR/TRAINING_DONE"
echo "✓ 训练完成: Explicit Risk"
echo "模型保存在: $OUTPUT_DIR"
echo "=========================================="

#!/bin/bash
ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
# Training script: Qwen3-4B-Instruct-Verbalized-CE (resume)
# Description: Verbalized Confidence + Cross-Entropy reward (Section 3.2.2)
# Usage examples:
#   bash rl/scripts/train_verbalized_ce_resume.sh
#   SAMPLE_SIZE=100 bash rl/scripts/train_verbalized_ce_resume.sh
#   USE_SAMPLE=0 TRAIN_FILES=/path/train.parquet VAL_FILES=/path/val.parquet bash rl/scripts/train_verbalized_ce_resume.sh
# tmux new -d -s ce_resume 'bash rl/scripts/train_verbalized_ce_resume.sh; code=$?; echo "[EXIT_CODE]=$code"; echo "[LAST_LOG]"; ls -t output/logs/verbalized_ce_resume_*.log 2>/dev/null | head -1 | xargs -r -I{} tail -n 120 {}; exec bash'

set -e

# Activate conda environment (supports different install paths)
if [ -f "/data1/conda/etc/profile.d/conda.sh" ]; then
    source /data1/conda/etc/profile.d/conda.sh
elif [ -f "/workspace/miniconda3/etc/profile.d/conda.sh" ]; then
    source /workspace/miniconda3/etc/profile.d/conda.sh
elif command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
else
    echo "conda not found. Please install conda or update the path in this script."
    exit 1
fi
conda deactivate 
conda activate truthrl-verl-npu

# Activate CANN environment if available
if [ -f "/usr/local/Ascend/ascend-toolkit/set_env.sh" ]; then
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi
if [ -f "/usr/local/Ascend/nnal/atb/set_env.sh" ]; then
    source /usr/local/Ascend/nnal/atb/set_env.sh
fi

# Configure HuggingFace mirror
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_CACHE=~/.cache/huggingface
export HF_HUB_OFFLINE=0
export TRANSFORMERS_OFFLINE=0
export HF_DATASETS_OFFLINE=0

# NPU & vLLM environment variables
export VLLM_DEVICE=npu
if [ -n "$ASCEND_RT_VISIBLE_DEVICES" ]; then
    export VERL_NPU_DEVICE_MAP="$ASCEND_RT_VISIBLE_DEVICES"
    NPU_COUNT=$(echo "$ASCEND_RT_VISIBLE_DEVICES" | awk -F',' '{print NF}')
    echo "[INFO] Using NPU devices: $ASCEND_RT_VISIBLE_DEVICES (total: $NPU_COUNT cards)"
    echo "[INFO] Device mapping: logical_rank -> physical_device = $VERL_NPU_DEVICE_MAP"
    unset ASCEND_RT_VISIBLE_DEVICES
elif [ -n "$NPU_VISIBLE_DEVICES" ]; then
    export VERL_NPU_DEVICE_MAP="$NPU_VISIBLE_DEVICES"
    NPU_COUNT=$(echo "$NPU_VISIBLE_DEVICES" | awk -F',' '{print NF}')
    echo "[INFO] Using NPU devices: $NPU_VISIBLE_DEVICES (total: $NPU_COUNT cards)"
    unset ASCEND_RT_VISIBLE_DEVICES
else
    NPU_COUNT="${NPU_COUNT:-8}"
    echo "[INFO] Using default NPU configuration: first $NPU_COUNT cards"
fi
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_ASCEND_ENABLE_NZ=0
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export WANDB_MODE=online
export WANDB_API_KEY="d0e4ffd3de59b61bc1bd9b069a15a76c4b3d9927"
export WANDB_RESUME="allow"
export WANDB_RUN_ID="7b92ycry"
export HCCL_PORT_RANGE=40000-40999
export HYDRA_FULL_ERROR=1
export VLLM_VERSION="${VLLM_VERSION:-0.11.0}"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"
export RAY_DEDUP_LOGS="${RAY_DEDUP_LOGS:-0}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export PYTHONPATH="$ROOT_DIR/rl/compat:${PYTHONPATH:-}"

USE_EXISTING_RAY_CLUSTER="${USE_EXISTING_RAY_CLUSTER:-0}"
if [ "$USE_EXISTING_RAY_CLUSTER" != "1" ]; then
    unset RAY_ADDRESS
    unset RAY_NAMESPACE
    echo "[INFO] Using local Ray runtime (unset RAY_ADDRESS/RAY_NAMESPACE)."
fi

if [ "$USE_EXISTING_RAY_CLUSTER" = "1" ]; then
    RAY_INIT_ADDRESS="${RAY_INIT_ADDRESS:-auto}"
else
    RAY_INIT_ADDRESS="${RAY_INIT_ADDRESS:-local}"
fi

# NPU stability toggles: disable torch compile by default to avoid known fused-kernel crashes
ACTOR_USE_TORCH_COMPILE="${ACTOR_USE_TORCH_COMPILE:-False}"
REF_USE_TORCH_COMPILE="${REF_USE_TORCH_COMPILE:-$ACTOR_USE_TORCH_COMPILE}"
REWARD_EPSILON="${REWARD_EPSILON:-0.01}"

SAMPLE_SIZE="${SAMPLE_SIZE:-}"
SAMPLE_SIZE_TRAIN="${SAMPLE_SIZE_TRAIN:-${SAMPLE_SIZE:-10000}}"
SAMPLE_SIZE_VAL="${SAMPLE_SIZE_VAL:-${SAMPLE_SIZE:-2000}}"
SAMPLE_SEED="${SAMPLE_SEED:-54}"
USE_SAMPLE="${USE_SAMPLE:-1}"
SRC_TRAIN="${SRC_TRAIN:-\
# $ROOT_DIR/verl/data/flash_rag/nq_train.parquet,\
$ROOT_DIR/verl/data/flash_rag/triviaqa_train.parquet,\
# $ROOT_DIR/verl/data/flash_rag/popqa_train.parquet,\
# $ROOT_DIR/verl/data/flash_rag/hotpotqa_train.parquet,\
# $ROOT_DIR/verl/data/flash_rag/2wikimultihopqa_train.parquet,\
# $ROOT_DIR/verl/data/flash_rag/musique_train.parquet,\
# $ROOT_DIR/verl/data/flash_rag/bamboogle_train.parquet\
}"

SRC_VAL="${SRC_VAL:-\
# $ROOT_DIR/verl/data/flash_rag/nq_test.parquet,\
$ROOT_DIR/verl/data/flash_rag/triviaqa_test.parquet,\
# $ROOT_DIR/verl/data/flash_rag/popqa_test.parquet,\
# $ROOT_DIR/verl/data/flash_rag/hotpotqa_test.parquet,\
# $ROOT_DIR/verl/data/flash_rag/2wikimultihopqa_test.parquet,\
# $ROOT_DIR/verl/data/flash_rag/musique_test.parquet,\
# $ROOT_DIR/verl/data/flash_rag/bamboogle_test.parquet\
}"
TRAIN_FILES="${TRAIN_FILES:-}"
VAL_FILES="${VAL_FILES:-}"

normalize_file_list() {
    local raw="$1"
    local result=""
    IFS=',' read -ra paths <<< "$raw"
    for path in "${paths[@]}"; do
        path="${path#"${path%%[![:space:]]*}"}"
        path="${path%"${path##*[![:space:]]}"}"
        if [ -n "$path" ]; then
            result="${result:+$result,}$path"
        fi
    done
    echo "$result"
}

to_hydra_list() {
    local csv
    csv="$(normalize_file_list "$1")"
    local out="["
    local first=1
    IFS=',' read -ra paths <<< "$csv"
    for path in "${paths[@]}"; do
        [ -z "$path" ] && continue
        if [ $first -eq 0 ]; then
            out+=" ,"
        fi
        out+="'$path'"
        first=0
    done
    out+="]"
    echo "$out"
}

LOG_DIR="$ROOT_DIR/output/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/resume_verbalized_ce_sample${SAMPLE_SIZE_TRAIN}_${SAMPLE_SIZE_VAL}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "[INFO] Log output: $LOG_FILE"

if [ -n "$TRAIN_FILES" ]; then
    SRC_TRAIN="$TRAIN_FILES"
fi
if [ -n "$VAL_FILES" ]; then
    SRC_VAL="$VAL_FILES"
fi

SRC_TRAIN="$(normalize_file_list "$SRC_TRAIN")"
SRC_VAL="$(normalize_file_list "$SRC_VAL")"

if [ "$USE_SAMPLE" != "0" ]; then
    TMP_DIR="/tmp/seedmitigating_sample${SAMPLE_SIZE_TRAIN}_${SAMPLE_SIZE_VAL}"
    mkdir -p "$TMP_DIR"
    TRAIN_FILES="$TMP_DIR/train_sample.parquet"
    VAL_FILES="$TMP_DIR/val_sample.parquet"
    export SRC_TRAIN SRC_VAL TRAIN_FILES VAL_FILES SAMPLE_SIZE_TRAIN SAMPLE_SIZE_VAL SAMPLE_SEED
    python3 - <<'PY'
import os
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np

src_train = os.environ.get("SRC_TRAIN", "")
src_val = os.environ.get("SRC_VAL", "")
out_train = os.environ.get("TRAIN_FILES")
out_val = os.environ.get("VAL_FILES")
sample_size_train = int(os.environ.get("SAMPLE_SIZE_TRAIN", "5000"))
sample_size_val = int(os.environ.get("SAMPLE_SIZE_VAL", "1000"))
sample_seed = int(os.environ.get("SAMPLE_SEED", "54"))


def parse_paths(csv):
    return [p.strip() for p in str(csv).split(",") if p and p.strip()]


def load_table(paths):
    if not paths:
        raise ValueError("No input parquet files provided")
    tables = []
    for path in paths:
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        tables.append(pq.read_table(path))
    return pa.concat_tables(tables) if len(tables) > 1 else tables[0]


def random_select_table(table, n, seed):
    total = table.num_rows
    n = max(1, min(n, total))
    if n == total:
        return table
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(total, size=n, replace=False))
    return table.take(pa.array(indices))


def build_sample(src_csv, out_path, n, seed):
    table = load_table(parse_paths(src_csv))
    table = random_select_table(table, n, seed)
    data = table.to_pydict()
    golden = data.get("golden_answers") or []
    data_source = data.get("data_source")
    if data_source is None:
        data["data_source"] = ["mixed_source"] * len(golden)
    reward_model = []
    for ga in golden:
        reward_model.append({"ground_truth": {"target": ga}, "style": "rule"})
    data["reward_model"] = reward_model
    out_table = pa.table(data)
    pq.write_table(out_table, out_path)


build_sample(src_train, out_train, sample_size_train, sample_seed)
build_sample(src_val, out_val, sample_size_val, sample_seed + 1)
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

TRAIN_FILES="$(normalize_file_list "$TRAIN_FILES")"
VAL_FILES="$(normalize_file_list "$VAL_FILES")"
if [ -z "$TRAIN_FILES" ] || [ -z "$VAL_FILES" ]; then
    echo "[ERROR] TRAIN_FILES/VAL_FILES is empty after normalization."
    exit 1
fi

HYDRA_TRAIN_FILES="$(to_hydra_list "$TRAIN_FILES")"
HYDRA_VAL_FILES="$(to_hydra_list "$VAL_FILES")"

echo "[INFO] Train files: $TRAIN_FILES"
echo "[INFO] Val files: $VAL_FILES"
echo "[INFO] Sample seed: $SAMPLE_SEED"

REWARD_FUNCTION_PATH="/root/code/SeedMitigating/rl/verl_custom_reward.py"
CUSTOM_DATASET_PATH="/root/code/SeedMitigating/rl/behavioral_dataset.py"
PROJECT_NAME="${PROJECT_NAME:-behavioral_calibration}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-verbalized_ce_triviaqa}"

RESUME_ROOT="/root/code/SeedMitigating/output/checkpoints/Qwen3-4B-Instruct-Verbalized-CE-20260307_235307"
RESUME_DIR="${RESUME_ROOT}/global_step_160"
OUTPUT_DIR="$RESUME_ROOT"
rm -f "$OUTPUT_DIR/TRAINING_DONE"
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "Training: Qwen3-4B-Instruct-Verbalized-CE"
echo "=========================================="
echo "Strategy: Verbalized Confidence + Cross-Entropy"
echo "Reward: CE (Paper Section 3.2.2)"
echo "Reward epsilon: $REWARD_EPSILON"
echo "Algorithm: GRPO"
echo "Resume from: $RESUME_DIR"
echo "Output: $OUTPUT_DIR"
echo ""

cd "$ROOT_DIR/verl"

python3 -m verl.trainer.main_ppo \
    --config-path=config \
    --config-name=ppo_trainer \
    algorithm.adv_estimator=grpo \
    algorithm.gamma=1.0 \
    algorithm.lam=1.0 \
    algorithm.norm_adv_by_std_in_grpo=True \
    data.train_files="$HYDRA_TRAIN_FILES" \
    data.val_files="$HYDRA_VAL_FILES" \
    data.train_batch_size=256 \
    data.max_prompt_length=512 \
    data.max_response_length=1536 \
    data.reward_fn_key=data_source \
    data.custom_cls.path="$CUSTOM_DATASET_PATH" \
    data.custom_cls.name=BehavioralCalibrationDataset \
    +data.strategy=verbalized_ce \
    actor_rollout_ref.model.path=Qwen/Qwen3-4B-Instruct-2507 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_torch_compile=$ACTOR_USE_TORCH_COMPILE \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.75 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.7 \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.use_torch_compile=$REF_USE_TORCH_COMPILE \
    algorithm.use_kl_in_reward=False \
    custom_reward_function.path="$REWARD_FUNCTION_PATH" \
    custom_reward_function.name=compute_score_verbalized_ce \
    +custom_reward_function.reward_kwargs.epsilon=$REWARD_EPSILON \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="$PROJECT_NAME" \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.n_gpus_per_node=$NPU_COUNT \
    trainer.nnodes=1 \
    trainer.save_freq=40 \
    trainer.test_freq=10 \
    trainer.total_epochs=30 \
    trainer.resume_mode=resume_path \
    trainer.resume_from_path="$RESUME_DIR" \
    trainer.device=npu \
    trainer.rollout_data_dir="$OUTPUT_DIR/rollout" \
    trainer.validation_data_dir="$OUTPUT_DIR/validation" \
    trainer.default_local_dir="$OUTPUT_DIR" \
    reward_model.enable=False \
    "$@"

echo ""
echo "=========================================="
touch "$OUTPUT_DIR/TRAINING_DONE"
echo "Training complete: Verbalized CE"
echo "Model saved to: $OUTPUT_DIR"
echo "=========================================="
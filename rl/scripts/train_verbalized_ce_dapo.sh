#!/bin/bash
ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
# Training script: Qwen3-4B-Instruct-Verbalized-CE on DAPO-Math-17k
# Eval on DAPO-Math (val) and BeyondAIME
# tmux new -d -s ce_dapo 'bash rl/scripts/train_verbalized_ce_dapo.sh; code=$?; echo "[EXIT_CODE]=$code"; echo "[LAST_LOG]"; ls -t output/logs/verbalized_ce_dapo_*.log 2>/dev/null | head -1 | xargs -r -I{} tail -n 120 {}; exec bash'

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

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HOME/.cache/huggingface}"
# Keep online by default because vLLM may query HF model metadata at startup.
# Users can still force offline mode by exporting these vars to 1.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-0}"

# Avoid broken localhost proxy inherited from shell that can crash vLLM tokenizer init.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
export NO_PROXY="${NO_PROXY:-127.0.0.1,localhost}"

export VLLM_DEVICE=npu
if [ -n "$ASCEND_RT_VISIBLE_DEVICES" ]; then
    export VERL_NPU_DEVICE_MAP="$ASCEND_RT_VISIBLE_DEVICES"
    NPU_COUNT=$(echo "$ASCEND_RT_VISIBLE_DEVICES" | awk -F',' '{print NF}')
    echo "[INFO] Using NPU devices: $ASCEND_RT_VISIBLE_DEVICES (total: $NPU_COUNT cards)"
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

if [ -z "$ROLLOUT_TP_SIZE" ]; then
    if [ "$NPU_COUNT" -eq 1 ]; then
        ROLLOUT_TP_SIZE=1
    else
        ROLLOUT_TP_SIZE=2
    fi
fi

export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_ASCEND_ENABLE_NZ=0
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export WANDB_MODE=online
export WANDB_API_KEY="d0e4ffd3de59b61bc1bd9b069a15a76c4b3d9927"
export HCCL_PORT_RANGE="${HCCL_PORT_RANGE:-40000-40999}"
export VLLM_VERSION="${VLLM_VERSION:-0.11.0}"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"
export RAY_DEDUP_LOGS="${RAY_DEDUP_LOGS:-0}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export WANDB_MODE="${WANDB_MODE:-disabled}"

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

ACTOR_USE_TORCH_COMPILE="${ACTOR_USE_TORCH_COMPILE:-False}"
REF_USE_TORCH_COMPILE="${REF_USE_TORCH_COMPILE:-$ACTOR_USE_TORCH_COMPILE}"
REWARD_EPSILON="${REWARD_EPSILON:-0.05}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-128}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-8}"
PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}"
REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-$ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}"
ROLLOUT_N="${ROLLOUT_N:-8}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
# 8x Ascend 910B stable profile: keep longer responses while avoiding logprob OOM.
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-6144}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.5}"
ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-1.0}"
ROLLOUT_TOP_P="${ROLLOUT_TOP_P:-1.0}"
ROLLOUT_TOP_K="${ROLLOUT_TOP_K:--1}"
VAL_TEMPERATURE="${VAL_TEMPERATURE:-1.0}"
VAL_TOP_P="${VAL_TOP_P:-0.7}"
VAL_TOP_K="${VAL_TOP_K:--1}"
VAL_DO_SAMPLE="${VAL_DO_SAMPLE:-False}"
ACTOR_LR="${ACTOR_LR:-1e-6}"
ACTOR_LR_WARMUP_STEPS="${ACTOR_LR_WARMUP_STEPS:-10}"
ACTOR_WEIGHT_DECAY="${ACTOR_WEIGHT_DECAY:-0.1}"
NORMALIZE_DAPO_PARQUET="${NORMALIZE_DAPO_PARQUET:-1}"

# Target training file
TRAIN_FILES="${TRAIN_FILES:-$ROOT_DIR/verl/data/dapo_math_train.parquet}"

# Evaluation files: DAPO-Math-Val and BeyondAIME
DAPO_VAL="${ROOT_DIR}/verl/data/dapo_math_val.parquet"
BEYONDAIME_VAL="${ROOT_DIR}/verl/data/beyondaime_val.parquet"

# Build BeyondAime val parquet if it doesn't exist (needed for combined eval)
BEYONDAIME_JSONL="${ROOT_DIR}/verl/data/beyondaime.jsonl"
PARQUET_CACHE_DIR="${PARQUET_CACHE_DIR:-/tmp/seedmitigating_beyondaime_parquet}"
mkdir -p "$PARQUET_CACHE_DIR"
AUTO_BEYONDAIME_VAL="$PARQUET_CACHE_DIR/beyondaime_val.parquet"
export BEYONDAIME_JSONL
export AUTO_BEYONDAIME_VAL

if [ ! -f "$BEYONDAIME_VAL" ] && [ ! -f "$AUTO_BEYONDAIME_VAL" ]; then
    echo "[INFO] Creating BeyondAime validation parquet from $BEYONDAIME_JSONL"
    python3 - <<'PY'
import json
import os
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

src_jsonl = os.environ.get("BEYONDAIME_JSONL", "../../verl/data/beyondaime.jsonl")
out_val = os.environ["AUTO_BEYONDAIME_VAL"]
train_ratio = 0.0
seed = 54

def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            rows.append(json.loads(line))
    return rows

def to_table(rows):
    questions, golden_answers, data_source, reward_model = [], [], [], []
    for row in rows:
        q = str(row.get("problem", "")).strip()
        a = str(row.get("answer", "")).strip()
        if not q or not a: continue
        questions.append(q)
        golden_answers.append([a])
        data_source.append("aime_2024")
        reward_model.append({"ground_truth": a, "style": "rule"})
    return pa.table({"question": questions, "golden_answers": golden_answers, "data_source": data_source, "reward_model": reward_model})

all_rows = load_jsonl(src_jsonl)
rng = np.random.default_rng(seed)
idx = np.arange(len(all_rows))
rng.shuffle(idx)
split = int(len(idx) * train_ratio)
val_rows = [all_rows[i] for i in idx[split:]]
val_table = to_table(val_rows)
pq.write_table(val_table, out_val)
print(f"[INFO] Built BeyondAIME val parquet: {out_val} ({val_table.num_rows} rows)")
PY
    BEYONDAIME_VAL="$AUTO_BEYONDAIME_VAL"
elif [ ! -f "$BEYONDAIME_VAL" ] && [ -f "$AUTO_BEYONDAIME_VAL" ]; then
    echo "[INFO] Using cached BeyondAIME val parquet: $AUTO_BEYONDAIME_VAL"
    BEYONDAIME_VAL="$AUTO_BEYONDAIME_VAL"
fi

if [ -z "$VAL_FILES" ]; then
    VAL_FILES="${DAPO_VAL},${BEYONDAIME_VAL}"
fi

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
    local csv="$(normalize_file_list "$1")"
    local out="["
    local first=1
    IFS=',' read -ra paths <<< "$csv"
    for path in "${paths[@]}"; do
        [ -z "$path" ] && continue
        [ $first -eq 0 ] && out+=","
        out+="'$path'"
        first=0
    done
    out+="]"
    echo "$out"
}

if [ "$NORMALIZE_DAPO_PARQUET" = "1" ]; then
    DAPO_PARQUET_CACHE_DIR="${DAPO_PARQUET_CACHE_DIR:-/tmp/seedmitigating_dapo_norm_parquet}"
    mkdir -p "$DAPO_PARQUET_CACHE_DIR"
    NORM_TRAIN="$DAPO_PARQUET_CACHE_DIR/train_norm.parquet"
    NORM_VAL="$DAPO_PARQUET_CACHE_DIR/val_norm.parquet"
    export TRAIN_FILES VAL_FILES NORM_TRAIN NORM_VAL
    echo "[INFO] Normalizing parquet schema for BehavioralCalibrationDataset..."
    python3 - <<'PY'
import os
from typing import Any
import pyarrow as pa
import pyarrow.parquet as pq


def parse_csv_paths(raw: str):
    return [p.strip() for p in str(raw).split(",") if p and p.strip()]


def extract_gt(v: Any):
    if v is None:
        return ""
    if isinstance(v, dict):
        if "target" in v:
            return extract_gt(v["target"])
        if "ground_truth" in v:
            return extract_gt(v["ground_truth"])
        if "answer" in v:
            return extract_gt(v["answer"])
        return ""
    if isinstance(v, (list, tuple)):
        if not v:
            return ""
        return extract_gt(v[0])
    s = str(v).strip()
    return s


def normalize_table(in_paths, out_path):
    out_q = []
    out_ga = []
    out_ds = []
    out_rm = []

    for path in in_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        table = pq.read_table(path)
        data = table.to_pydict()
        n = table.num_rows
        question = data.get("question")
        prompt = data.get("prompt")
        golden_answers = data.get("golden_answers")
        data_source = data.get("data_source")
        reward_model = data.get("reward_model")
        reward_model_gt_flat = data.get("reward_model.ground_truth")

        for i in range(n):
            q = ""
            if question is not None and i < len(question) and question[i] is not None:
                q = str(question[i]).strip()
            elif prompt is not None and i < len(prompt) and prompt[i] is not None:
                q = str(prompt[i]).strip()

            gt = ""
            if reward_model is not None and i < len(reward_model):
                gt = extract_gt(reward_model[i])
            if not gt and reward_model_gt_flat is not None and i < len(reward_model_gt_flat):
                gt = extract_gt(reward_model_gt_flat[i])
            if not gt and golden_answers is not None and i < len(golden_answers):
                gt = extract_gt(golden_answers[i])

            ds = "dapo_math"
            if data_source is not None and i < len(data_source) and data_source[i] is not None:
                ds_raw = str(data_source[i]).strip()
                ds = ds_raw if ds_raw else ds

            out_q.append(q)
            out_ga.append([gt] if gt else [])
            out_ds.append(ds)
            out_rm.append({"ground_truth": gt, "style": "rule"})

    out = pa.table(
        {
            "question": out_q,
            "golden_answers": out_ga,
            "data_source": out_ds,
            "reward_model": out_rm,
        }
    )
    pq.write_table(out, out_path)
    n_empty_gt = sum(1 for x in out_ga if not x)
    print(f"[INFO] Normalized {out_path}: rows={out.num_rows}, empty_gt={n_empty_gt}")


train_paths = parse_csv_paths(os.environ["TRAIN_FILES"])
val_paths = parse_csv_paths(os.environ["VAL_FILES"])
normalize_table(train_paths, os.environ["NORM_TRAIN"])
normalize_table(val_paths, os.environ["NORM_VAL"])
PY
    TRAIN_FILES="$NORM_TRAIN"
    VAL_FILES="$NORM_VAL"
fi

HYDRA_TRAIN_FILES="$(to_hydra_list "$TRAIN_FILES")"
HYDRA_VAL_FILES="$(to_hydra_list "$VAL_FILES")"

LOG_DIR="$ROOT_DIR/output/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/verbalized_ce_dapo_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "[INFO] Train files: $TRAIN_FILES"
echo "[INFO] Val files: $VAL_FILES"
echo "[INFO] Rollout log_prob micro batch size / GPU: $ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU"
echo "[INFO] Ref log_prob micro batch size / GPU: $REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU"

REWARD_FUNCTION_PATH="$ROOT_DIR/rl/verl_custom_reward.py"
CUSTOM_DATASET_PATH="$ROOT_DIR/rl/behavioral_dataset.py"
PROJECT_NAME="${PROJECT_NAME:-behavioral_calibration}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-verbalized_ce_dapo}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/output/checkpoints}"
OUTPUT_DIR="$OUTPUT_DIR/Qwen3-4B-Instruct-Verbalized-CE-DAPO-$(date +%Y%m%d_%H%M%S)"

mkdir -p "$OUTPUT_DIR"

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
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    data.truncation=left \
    data.reward_fn_key=data_source \
    data.custom_cls.path="$CUSTOM_DATASET_PATH" \
    data.custom_cls.name=BehavioralCalibrationDataset \
    +data.remove_math_datasets=False \
    +data.strategy=verbalized_ce \
    actor_rollout_ref.model.path=Qwen/Qwen3-4B-Instruct-2507 \
    actor_rollout_ref.actor.optim.lr=$ACTOR_LR \
    actor_rollout_ref.actor.optim.lr_warmup_steps=$ACTOR_LR_WARMUP_STEPS \
    actor_rollout_ref.actor.optim.weight_decay=$ACTOR_WEIGHT_DECAY \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_SIZE_PER_GPU \
    actor_rollout_ref.actor.use_torch_compile=$ACTOR_USE_TORCH_COMPILE \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=$ROLLOUT_GPU_MEMORY_UTILIZATION \
    actor_rollout_ref.rollout.n=$ROLLOUT_N \
    actor_rollout_ref.rollout.temperature=$ROLLOUT_TEMPERATURE \
    actor_rollout_ref.rollout.top_p=$ROLLOUT_TOP_P \
    actor_rollout_ref.rollout.top_k=$ROLLOUT_TOP_K \
    actor_rollout_ref.rollout.response_length=$MAX_RESPONSE_LENGTH \
    actor_rollout_ref.rollout.val_kwargs.temperature=$VAL_TEMPERATURE \
    actor_rollout_ref.rollout.val_kwargs.top_p=$VAL_TOP_P \
    actor_rollout_ref.rollout.val_kwargs.top_k=$VAL_TOP_K \
    actor_rollout_ref.rollout.val_kwargs.do_sample=$VAL_DO_SAMPLE \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP_SIZE \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU \
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
    trainer.test_freq=5 \
    trainer.total_epochs=10 \
    trainer.device=npu \
    trainer.rollout_data_dir="$OUTPUT_DIR/rollout" \
    trainer.validation_data_dir="$OUTPUT_DIR/validation" \
    trainer.default_local_dir="$OUTPUT_DIR" \
    reward_model.enable=False \
    +ray_kwargs.ray_init.address="$RAY_INIT_ADDRESS" \
    "$@"

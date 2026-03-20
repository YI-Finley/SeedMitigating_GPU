# Mitigating LLM Hallucination via Behaviorally Calibrated RL — 一站式复现/对照指南 (Ascend NPU)

本仓库用于复现论文《Mitigating LLM Hallucination via Behaviorally Calibrated Reinforcement Learning》全部实验。

## 0. 必读说明

- 仓库路径已规范为 `/root/SeedMitigating`（无末尾空格）。
- `/root/SeedMitigating/behavioral_calibration.py` 为作者提供的核心实现，视为**权威且不可修改**。
- 允许偏差：**外部模型集合、数据规模、超参设置**；除此之外必须与论文一致。
- 输出目录可通过 `SEEDMIT_OUTPUT_BASE` 或 `OUTPUT_BASE` 指定；默认优先 `/data2/SeedMitigating-output/paper_reproduction`，否则使用 `output/paper_reproduction`。

## 1. 论文 ↔ 代码对照（方法 / 公式 / 数据）

### Section 3 方法（Methodology）

- **3.1 Behavioral Calibration 四目标** → `eval/behavioral_calibration_metrics.py` + `eval/visualize.py`
  - Adaptive Risk / Accuracy Preservation / Hallucination Reduction / Quantitative Calibration。
- **Eq.(1) SNR-Gain** → `eval/confidence_calibration_metrics.py`（表格指标）与 `eval/behavioral_calibration_metrics.py`（曲线）。
- **3.2.1 Explicit Risk Thresholding (Eq.(2))**
  - Prompt：`rl/behavioral_dataset.py`
  - 训练：`rl/scripts/train_explicit_risk.sh`
  - 评估：`rl/scripts/eval_explicit_risk.sh`
  - 图1：`rl/evaluation/generate_figure1.py`
- **3.2.2 Verbalized Confidence（Eq.(3)/(4)）**
  - 权威奖励：`behavioral_calibration.py`
  - 同步实现：`rl/rewards.py`
  - VERL 接入：`rl/verl_custom_reward.py`
  - Prompt：`rl/behavioral_dataset.py`
- **3.2.3 Critic Value**
  - 训练/评估入口：`rl/scripts/train_ppo_value.sh` + `scripts/evaluate_model_fixed.py`
  - value→confidence 映射：`rl/verl_custom_reward.py`
- **3.3 Claim-level Calibration**
  - Claim HTML 格式与 `<Confidence value=...>`：`rl/behavioral_dataset.py`
  - 聚合（Product / Minimum）：`behavioral_calibration.py`
  - Claim 标注：`scripts/label_claims_noapi.py`

### Section 4 实验（Experiments）

- **4.1 Training Setup** → `rl/scripts/train_*` + `run_full_training*.sh`（训练数据：`data/dapo_math_train.parquet`）
- **4.2 Response-level Evaluation** → `scripts/evaluate_model_fixed.py` + `rl/scripts/eval_response_level_*.sh`
- **4.2 Claim-level Evaluation** → `rl/scripts/eval_claim_level_beyondaime.sh` + `scripts/label_claims_noapi.py`
- **4.3 Behavioral Criteria** → `eval/behavioral_calibration_metrics.py` + `rl/evaluation/generate_figure6.py` + `generate_figure7.py`
- **4.4 SimpleQA** → `rl/scripts/eval_simpleqa.sh` + `eval/simpleqa_grader.py`（官方 prompt） + `generate_table4.py`
- **4.5 Test-Time Scaling** → `run_test_time_scaling.sh` + `scripts/evaluate_test_time_scaling.py` + `generate_figure8.py`

## 2. 图表 / 表格 对照表（论文 ↔ 生成脚本）

### 图（Figure）

| 图 | 原文位置 | 生成方式 | 依赖输入 | 输出文件 |
|---|---|---|---|---|
| Figure 1 | Sec. 3.2.1 | `rl/evaluation/generate_figure1.py` | `rl/scripts/eval_explicit_risk.sh` 产生的 AIME-2024 结果 | `output*/figures/figure1_explicit_risk_progress.png` |
| Figure 2 | Sec. 3.3.1 | **人工选样**（无自动脚本） | `scripts/evaluate_model_fixed.py` 的 `results.json` 中挑选 `<Confidence>` 示例 | 手工插图 |
| Figure 3 | Sec. 3.3.2 | **需要额外导出 critic token-level value**（当前无自动脚本） | 在 PPO-Value 评估中导出 token-level value 并可视化 | 手工插图 |
| Figure 4 | Sec. 4.2.1 | `rl/evaluation/generate_figure4.py` | `rl/scripts/eval_response_level_beyondaime.sh` + （可选）`rl/scripts/eval_frontier_noapi.sh` | `output*/figures/figure4_response_level_calibration.png` |
| Figure 5 | Sec. 4.2.2 | `rl/evaluation/generate_figure5.py --labels_file ...` | `rl/scripts/eval_claim_level_beyondaime.sh` + claim 标注文件 | `output*/figures/figure5_claim_level_calibration.png` |
| Figure 6 | Sec. 4.3 | `rl/evaluation/generate_figure6.py [--labels_file ...]` | response/claim-level 评估结果 +（可选）claim 标注 | `output*/figures/figure6_adaptive_risk.png` |
| Figure 7 | Sec. 4.3 | `rl/evaluation/generate_figure7.py [--labels_file ...]` | response/claim-level 评估结果 +（可选）claim 标注 | `output*/figures/figure7_quantitative_calibration.png` |
| Figure 8 | Sec. 4.5 | `rl/evaluation/generate_figure8.py` | `run_test_time_scaling.sh` 产生的 TTS 结果 | `output*/figures/figure8_test_time_scaling.png` |

> `output*` 表示 `output/paper_reproduction` 或 `/data2/SeedMitigating-output/paper_reproduction`，由 `output_utils.get_output_base()` 自动判定。

### 表（Table）

| 表 | 原文位置 | 生成脚本 | 依赖输入 | 输出文件 |
|---|---|---|---|---|
| Table 1 | Sec. 4.2.1 | `rl/evaluation/generate_table1.py` | `rl/scripts/eval_response_level_beyondaime.sh` +（可选）前沿模型结果 | `output*/tables/table1_response_level_beyondaime.csv` |
| Table 2 | Sec. 4.2.1 | `rl/evaluation/generate_table2.py` | `rl/scripts/eval_response_level_aime.sh` | `output*/tables/table2_response_level_aime.csv` |
| Table 3 | Sec. 4.2.2 | `rl/evaluation/generate_table3.py --labels_file ...` | claim-level 结果 + claim 标注文件 | `output*/tables/table3_claim_level_beyondaime.csv` |
| Table 4 | Sec. 4.4 | `rl/evaluation/generate_table4.py` | `rl/scripts/eval_simpleqa.sh`（含 SimpleQA 官方评分） | `output*/tables/table4_response_level_simpleqa.csv` |

## 3. 最小复现流程（全链路）

```bash
# 1) 环境与依赖
pip install -r "/root/SeedMitigating/requirements-npu.txt"

# 2) 训练（6个主变体）
bash "/root/SeedMitigating/rl/scripts/train_all_variants.sh"

# 3) 评估（响应级）
bash "/root/SeedMitigating/rl/scripts/eval_response_level_beyondaime.sh"
bash "/root/SeedMitigating/rl/scripts/eval_response_level_aime.sh"

# 4) Claim-level（需要外部 LLM 标注）
export OPENAI_API_BASE=https://noapi.ggb.today/v1
export OPENAI_API_KEY=你的key
export NOAPI_LABEL_MODEL=gpt-4o-mini
bash "/root/SeedMitigating/rl/scripts/eval_claim_level_beyondaime.sh"

# 5) SimpleQA（官方 judge）
bash "/root/SeedMitigating/rl/scripts/eval_simpleqa.sh"

# 6) 生成表格与图
python "/root/SeedMitigating/rl/evaluation/generate_table1.py"
python "/root/SeedMitigating/rl/evaluation/generate_table2.py"
python "/root/SeedMitigating/rl/evaluation/generate_table3.py" --labels_file /path/to/claim_labels.jsonl
python "/root/SeedMitigating/rl/evaluation/generate_table4.py"
python "/root/SeedMitigating/rl/evaluation/generate_figure1.py"
python "/root/SeedMitigating/rl/evaluation/generate_figure4.py"
python "/root/SeedMitigating/rl/evaluation/generate_figure5.py" --labels_file /path/to/claim_labels.jsonl
python "/root/SeedMitigating/rl/evaluation/generate_figure6.py" --labels_file /path/to/claim_labels.jsonl
python "/root/SeedMitigating/rl/evaluation/generate_figure7.py" --labels_file /path/to/claim_labels.jsonl
python "/root/SeedMitigating/rl/evaluation/generate_figure8.py"
```

## 4. SimpleQA 官方评分

- 官方评分 prompt：`eval/simpleqa_grader.py`（来自 openai/simple-evals 逻辑）。
- 评估脚本：`scripts/evaluate_model_fixed.py --simpleqa_grader gpt-4o-mini`。
- 只要 `OPENAI_API_BASE`/`OPENAI_API_KEY` 可用，脚本会调用 LLM judge；否则回退到字符串匹配（不推荐）。

## 5. 文件索引（逐文件说明）

> 说明：本索引**逐文件**覆盖复现相关内容；`verl/` 为第三方训练框架（文件量巨大），按目录说明。

### 根目录

- `behavioral_calibration.py`：作者提供的核心奖励/判题/置信度逻辑（权威实现）。
- `Mitigating LLM Hallucination via Behaviorally Calibrated Reinforcement Learning.pdf`：论文原文 PDF。
- `paper.txt`：论文文本抽取版（便于搜索）。
- `README.md`：本指南。
- `REVIEW_GUIDE.md`：逐节对照 review 清单（更细化的核对步骤）。
- `RUN_PLAN.md`：运行计划与阶段性安排。
- `requirements.txt`：基础依赖。
- `requirements-npu.txt`：昇腾 NPU 依赖。
- `setup_npu_env.sh`：NPU 环境初始化脚本。
- `run_full_training.sh`：全量训练入口（默认全数据）。
- `run_full_training_1k.sh`：1k 子集训练入口（快速验证）。
- `run_full_training_5k.sh`：5k 子集训练入口（中等规模）。
- `run_final_training.sh`：历史训练入口（保留）。
- `run_evaluation.sh`：统一评估入口（串联评估脚本）。
- `run_parallel_beyondaime.sh`：BeyondAIME 并行评估入口。
- `run_test_time_scaling.sh`：Test-Time Scaling 实验入口。
- `data/`：数据集与数据工具。
- `eval/`：评估指标与可视化。
- `scripts/`：评估与标注脚本。
- `rl/`：论文复现核心实现（奖励、prompt、脚本、图表）。
- `output/`：本地运行产物（可清理或复用）。
- `verl/`：第三方训练框架（VERL，上游工程）。
- `.claude/`：工具生成元数据（非核心）。
- `__pycache__/`：Python 缓存（非核心）。

### data/

- `aime_2024.jsonl`：AIME-2024 评估集。
- `aime_2025.jsonl`：AIME-2025 评估集。
- `beyondaime.jsonl`：BeyondAIME 评估集（主测）。
- `simpleqa.jsonl`：SimpleQA 评估集（跨域）。
- `gsm8k_train.jsonl`：备用/对比数据（非论文主实验）。
- `dapo_math_train.parquet`：DAPO-Math-17k 训练集。
- `dapo_math_val.parquet`：DAPO-Math 验证集。
- `dapo_math_train_1k.parquet`：1k 训练子集（冒烟/快速）。
- `dapo_math_train_5k.parquet`：5k 训练子集（中等规模）。
- `debug_train_8.parquet` / `debug_train_32.parquet`：小样本调试训练集。
- `debug_val_8.parquet` / `debug_val_32.parquet`：小样本调试验证集。
- `dapo_math_loader.py`：DAPO 数据加载工具。

### eval/

- `confidence_calibration_metrics.py`：smECE/Brier/NLL/ConfAUC/SNR-Gain/AbsAcc/PredAcc。
- `behavioral_calibration_metrics.py`：Adaptive Risk / Accuracy Preservation / Hallucination Reduction / Quantitative Calibration。
- `simpleqa_grader.py`：SimpleQA 官方评分 prompt 与判分逻辑。
- `visualize.py`：图6/图7 可视化工具（与论文图形一致）。

### scripts/

- `evaluate_model_fixed.py`：主评估脚本（生成 results.json，调用判题与指标计算）。
- `evaluate_external_model.py`：外部模型（OpenAI 兼容 API）评估。
- `evaluate_test_time_scaling.py`：TTS 评估与汇总。
- `label_claims_noapi.py`：使用外部 LLM 标注 claim-level 正确性。

### rl/

- `behavioral_dataset.py`：训练/评估 prompt 模板（Response-level / Explicit Risk / Claim-level）。
- `rewards.py`：奖励函数实现（对齐 `behavioral_calibration.py`）。
- `verl_custom_reward.py`：VERL 奖励接口封装。
- `convert_dataset_to_parquet.py`：一次性数据转换脚本（DAPO → parquet）。
- `QUICK_START.md`：快速上手指南。
- `README.md`：RL 目录说明。
- `REPRODUCTION_GUIDE.md`：论文第4节完整复现指南。
- `SCRIPTS_INDEX.md`：训练与评估脚本索引。
- `scripts/`：训练与评估入口脚本（见下）。
- `evaluation/`：表格/图表生成脚本（见下）。

### rl/scripts/

- `train_all_variants.sh`：训练 6 个主变体（串/并）。
- `train_baseline_ppo.sh`：Baseline PPO。
- `train_confidence_brier.sh`：Confidence-Brier。
- `train_confidence_ce.sh`：Confidence-CE（GRPO）。
- `train_ppo_value.sh`：PPO-Value。
- `train_confidence_prod.sh`：Claim-level Product 聚合。
- `train_confidence_min.sh`：Claim-level Minimum 聚合。
- `train_explicit_risk.sh`：Explicit Risk Thresholding。
- `eval_response_level_beyondaime.sh`：BeyondAIME 响应级评估。
- `eval_response_level_aime.sh`：AIME 评估。
- `eval_simpleqa.sh`：SimpleQA 评估（含 LLM judge）。
- `eval_claim_level_beyondaime.sh`：claim-level 评估与标注流程。
- `eval_adaptive_risk.sh`：Adaptive Risk 曲线评估。
- `eval_quantitative_calibration.sh`：TP/FN 曲线评估。
- `eval_explicit_risk.sh`：Explicit Risk 评估（用于图1）。
- `eval_test_time_scaling.sh`：TTS 评估。
- `eval_frontier_noapi.sh`：外部模型 noapi 评估入口。
- `generate_all_results.sh`：一键生成所有表/图。

### rl/evaluation/

- `output_utils.py`：输出目录选择逻辑。
- `generate_table1.py`：Table 1（BeyondAIME 响应级）。
- `generate_table2.py`：Table 2（AIME 2024/2025）。
- `generate_table3.py`：Table 3（BeyondAIME 声明级）。
- `generate_table4.py`：Table 4（SimpleQA）。
- `generate_figure1.py`：Figure 1（Explicit Risk 训练进度）。
- `generate_figure4.py`：Figure 4（响应级校准图）。
- `generate_figure5.py`：Figure 5（声明级校准图）。
- `generate_figure6.py`：Figure 6（Adaptive Risk 面积图）。
- `generate_figure7.py`：Figure 7（TP/FN 曲线）。
- `generate_figure8.py`：Figure 8（TTS 曲线）。

### output/

- `output/paper_reproduction/`：运行产物（表格/图/日志）。
  - `tables/table1_response_level_beyondaime.csv`
  - `tables/table2_response_level_aime.csv`
  - `test_time_scaling_smoke/*/test_time_scaling.log`
- `output/section4_final_training/`：历史训练日志/模型（非必要）。

### verl/（第三方框架）

- 这是上游 VERL 训练框架，文件量巨大且不属于本文复现逻辑主体。
- 复现只通过 `rl/` 中的封装脚本使用其能力；如需深入审查，请从 `verl/README.md` 开始。

---

如需逐条审查，建议先按 `REVIEW_GUIDE.md` 执行论文对照检查，再结合本 README 的“图表对照表 + 文件索引”。

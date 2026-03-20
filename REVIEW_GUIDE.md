# 论文对照式代码 Review 指南（Mitigating LLM Hallucination via Behaviorally Calibrated RL）

> 目的：按论文逐节核对本仓库实现（除外部模型集合 / 数据规模 / 超参设置之外，其他必须一致）。
> 注意：`/root/SeedMitigating/behavioral_calibration.py` 为作者核心文件，视为权威实现，不应修改。

## 0. 先读论文，再对照代码
建议顺序：
1) Section 3 Methodology（式(1)(3)(4)、策略定义、claim-level）
2) Section 4 Experiments（Table/ Figure 1–8 定义与数据来源）
3) SimpleQA 评测细节与判分标准（官方 SimpleQA judge）

仓库内论文文本：`/root/SeedMitigating/paper.txt`
论文 PDF：`/root/SeedMitigating/Mitigating LLM Hallucination via Behaviorally Calibrated Reinforcement Learning.pdf`

---

## 1. 快速索引：论文 → 代码
### 方法与奖励
- **Explicit Risk Thresholding**（3.2.1）
  - Prompt/数据封装：`/root/SeedMitigating/rl/behavioral_dataset.py`
  - 训练入口：`/root/SeedMitigating/rl/scripts/train_explicit_risk.sh`
  - 评估/图1：`/root/SeedMitigating/rl/scripts/eval_explicit_risk.sh` + `rl/evaluation/generate_figure1.py`
- **Verbalized Confidence (Brier)**（Eq.3）
  - 权威实现：`/root/SeedMitigating/behavioral_calibration.py`（reward_func_confidence_brier）
  - 训练入口：`/root/SeedMitigating/rl/scripts/train_confidence_brier.sh`
- **Verbalized Confidence (CE)**（Eq.4）
  - 权威实现：`/root/SeedMitigating/behavioral_calibration.py`（reward_func_confidence_ce）
  - 训练入口：`/root/SeedMitigating/rl/scripts/train_confidence_ce.sh`
- **Critic Value**（3.2.3）
  - 训练入口：`/root/SeedMitigating/rl/scripts/train_ppo_value.sh`
  - 评分映射：`/root/SeedMitigating/rl/verl_custom_reward.py`（value→confidence）
- **Claim-level Calibration**（3.3）
  - Prompt 模板：`/root/SeedMitigating/rl/behavioral_dataset.py`
  - 聚合：`behavioral_calibration.py`（product/min）
  - 评估/标注：`/root/SeedMitigating/scripts/label_claims_noapi.py`

### 评估指标与表格
- 指标实现（smECE/Brier/NLL/AUC/SNR-Gain/AbsAcc/PredAcc）：`/root/SeedMitigating/eval/confidence_calibration_metrics.py`
- 行为校准目标（TP/FN/Adaptive Risk 等）：`/root/SeedMitigating/eval/behavioral_calibration_metrics.py`
- Table 1：`/root/SeedMitigating/rl/evaluation/generate_table1.py`
- Table 2：`/root/SeedMitigating/rl/evaluation/generate_table2.py`
- Table 3：`/root/SeedMitigating/rl/evaluation/generate_table3.py`
- Table 4：`/root/SeedMitigating/rl/evaluation/generate_table4.py`

### 图表脚本
- Figure 1：`rl/evaluation/generate_figure1.py`
- Figure 4：`rl/evaluation/generate_figure4.py`
- Figure 5：`rl/evaluation/generate_figure5.py`
- Figure 6：`rl/evaluation/generate_figure6.py`
- Figure 7：`rl/evaluation/generate_figure7.py`
- Figure 8：`rl/evaluation/generate_figure8.py`

### SimpleQA 官方判分
- 官方判分 prompt（openai/simple-evals）：`/root/SeedMitigating/eval/simpleqa_grader.py`
- 使用方式：`scripts/evaluate_model_fixed.py --simpleqa_grader gpt-4o-mini`

---

## 2. 核心一致性 Checklist（逐条对照）
### 2.1 模型与数据
- 基础模型：`Qwen/Qwen3-4B-Instruct-2507`
- 训练数据：`/root/SeedMitigating/data/dapo_math_train.parquet`
- AIME/BeyondAIME/SimpleQA：`/root/SeedMitigating/data/*.jsonl`

### 2.2 Prompt 形式
- **Response-level**：
  - `Answer: ...` + `Confidence: ...`
- **Explicit Risk**：
  - “Answer only if you are > t confident ... <IDK> ...”
- **Claim-level**：
  - `<begin_solution>` + `<Confidence value=...>Step ...` + `<end_solution>`

### 2.3 公式实现（逐式核对）
- **Eq.(1) SNR Gain**：`eval/confidence_calibration_metrics.py`
  - `SNR([0,1]) = ∫Acc / ∫Hal` + `log(SNR([0,1]) / SNR(0))`
- **Eq.(3) Brier Reward**：`behavioral_calibration.py` / `rl/rewards.py`
  - `R = 2p·valid(y) - p^2`
- **Eq.(4) CE Reward**：`behavioral_calibration.py`
  - 分母 `log((1-ε)/ε)`，`p' = clip(p, ε, 1-ε)`

### 2.4 评估判题
- 数学类：`behavioral_calibration.py` 的 Minerva 判题
- SimpleQA：**必须使用官方 LLM judge prompt**（`eval/simpleqa_grader.py`）

### 2.5 Claim-level 标注
- Judge 输入必须包含：问题 + 完整解答（含 `<Confidence>` 标签）+ 最终答案真值标签
- 输出：`[0/1, ...]` 与 claim 顺序严格对齐

---

## 3. 推荐的 Review 步骤（逐步对照）
1) **核对论文公式** → 对照 `behavioral_calibration.py` 与 `rl/rewards.py`
2) **核对 prompt 设计** → `rl/behavioral_dataset.py`
3) **核对评估指标** → `eval/confidence_calibration_metrics.py` / `eval/behavioral_calibration_metrics.py`
4) **核对评估流程** → `scripts/evaluate_model_fixed.py` / `scripts/evaluate_external_model.py`
5) **核对表图生成** → `rl/evaluation/generate_table*.py` / `generate_figure*.py`
6) **核对 SimpleQA 判分** → `eval/simpleqa_grader.py` + 实际跑通日志
7) **核对 claim-level 标注** → `scripts/label_claims_noapi.py` 的输入/输出格式

---

## 4. 冒烟验证（可选）
### 小样本跑通（响应级）
```bash
SMOKE_SAMPLES=1 bash "/root/SeedMitigating/run_evaluation_smoke.sh"
```
- 只验证流程可跑通，不要求数值对齐论文。
- 如果已有结果，会自动跳过；可先删除对应 `results.json` 强制重跑。

### 外部模型（仅 gpt-4o-mini）
```bash
export OPENAI_API_BASE=https://noapi.ggb.today/v1
export OPENAI_API_KEY=***
bash "/root/SeedMitigating/rl/scripts/eval_frontier_noapi.sh"
```

---

## 5. 常见问题与自检
- **noapi 503**：SimpleQA 判分与 claim 标注会回退失败；必须保证 `/chat/completions` 可用。
- **离线 HF**：设置 `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1` 避免联网失败。
- **AIME 字段兼容**：`Problem/Answer` 已在评估脚本中支持。
- **claim 标签缺失**：Table 3 / Figure 5/6/7 需要 claim-level labels。

---

## 6. 允许偏差范围（按你的要求）
- 外部模型集合
- 数据规模
- 超参设置

除此之外的内容必须严格对齐论文。

---

## 7. 当前仓库状态提示
- 评估流程可以跑通，但会较慢。
- 若你看到 `results.json` 缺失，可直接重跑冒烟脚本生成。
- SimpleQA 官方判分需要 noapi 可用，否则会回退到字符串匹配。


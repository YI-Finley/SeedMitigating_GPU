# 论文第4节完整复现指南

本文档提供《Mitigating LLM Hallucination via Behaviorally Calibrated Reinforcement Learning》第4节所有实验的完整复现方案。

## 环境配置

- **虚拟环境**: `truthrl-verl-npu`
- **基础模型**: Qwen3-4B-Instruct-2507
- **训练框架**: VERL (NPU适配版本)
- **数据集**: DAPO-Math-17k, BeyondAIME, AIME-2024/2025, SimpleQA

## 实验结构

```
第4节实验复现
├── 4.1 训练配置 (Training Setup)
│   ├── 训练6个模型变体
│   └── 准备前沿模型对比
├── 4.2 置信度校准定量评估 (Quantitative Evaluation)
│   ├── 4.2.1 评估指标实现
│   ├── 4.2.2 响应级评估 (BeyondAIME)
│   └── 4.2.3 声明级评估 (BeyondAIME)
├── 4.3 行为校准目标验证 (Behavioral Calibration Criteria)
│   ├── Adaptive Risk
│   ├── Accuracy Preservation
│   ├── Hallucination Reduction
│   └── Quantitative Calibration
├── 4.4 跨域迁移评估 (Direct Transfer to Factual QA)
│   └── SimpleQA评估
└── 4.5 测试时缩放 (Test-Time Scaling)
    └── 多轮预测聚合实验
```

## 4.1 训练配置 (Training Setup)

### 4.1.1 模型变体训练脚本

#### 变体1: Qwen3-4B-Instruct-PPO (Baseline)
**脚本**: `train_baseline_ppo.sh`
**描述**: 标准PPO训练，使用二元正确性奖励
```bash
bash /root/SeedMitigating\ /rl/scripts/train_baseline_ppo.sh
```

#### 变体2: Qwen3-4B-Instruct-Confidence-Brier
**脚本**: `train_confidence_brier.sh`
**描述**: Verbalized Confidence + Brier Score (均匀分布风险阈值)
```bash
bash /root/SeedMitigating\ /rl/scripts/train_confidence_brier.sh
```

#### 变体3: Qwen3-4B-Instruct-Confidence-CE
**脚本**: `train_confidence_ce.sh`
**描述**: Verbalized Confidence + Cross-Entropy (Beta分布风险阈值，使用GRPO)
```bash
bash /root/SeedMitigating\ /rl/scripts/train_confidence_ce.sh
```

#### 变体4: Qwen3-4B-Instruct-PPO-Value
**脚本**: `train_ppo_value.sh`
**描述**: PPO Critic网络输出作为置信度
```bash
bash /root/SeedMitigating\ /rl/scripts/train_ppo_value.sh
```

#### 变体5: Qwen3-4B-Instruct-Confidence-Prod
**脚本**: `train_confidence_prod.sh`
**描述**: 声明级置信度 + 乘积聚合
```bash
bash /root/SeedMitigating\ /rl/scripts/train_confidence_prod.sh
```

#### 变体6: Qwen3-4B-Instruct-Confidence-Min
**脚本**: `train_confidence_min.sh`
**描述**: 声明级置信度 + 最小值聚合
```bash
bash /root/SeedMitigating\ /rl/scripts/train_confidence_min.sh
```

### 4.1.2 训练参数配置

所有模型使用统一的训练配置（与论文保持一致）：

| 参数 | 值 | 说明 |
|------|-----|------|
| 基础模型 | Qwen3-4B-Instruct-2507 | 必须使用该模型 |
| 训练数据 | DAPO-Math-17k | 17k数学问题 |
| 训练轮数 | 30 epochs | 论文标准配置 |
| 学习率 | 1e-5 | Actor和Critic |
| Batch Size | 128 | 全局batch size |
| γ (gamma) | 0.99 | GAE折扣因子 |
| λ (lambda) | 0.95 | GAE权衡参数 |
| 温度 | 0.7 | 采样温度 |
| Top-p | 0.9 | Nucleus采样 |
| Max Response Length | 2048 | 最大生成长度 |

### 4.1.3 评估基础设施

本复现使用已有的评估代码：
- **置信度校准指标**: `/root/SeedMitigating/eval/confidence_calibration_metrics.py`
- **行为校准指标**: `/root/SeedMitigating/eval/behavioral_calibration_metrics.py`
- **可视化模块**: `/root/SeedMitigating/eval/visualize.py`
- **评估脚本**: `/root/SeedMitigating/scripts/evaluate_model_fixed.py`

**注意**: 已支持外部模型（noapi，便宜模型）对比评估，默认包含 `gpt-4o-mini`、`gemini-2.5-flash`、`zai-org/glm-4.5`、`grok-3`、`deepseek/deepseek-v3-0324`、`虹猫大模型-o3`。

## 4.2 置信度校准定量评估

### 4.2.1 评估指标实现

**脚本**: `evaluation/metrics.py`

实现以下指标：
1. **Smoothed ECE (smECE)**: 使用核密度估计的期望校准误差
2. **Brier Score**: 置信度与真实结果的均方误差
3. **Negative Log-Likelihood (NLL)**: 对数似然损失
4. **Confidence AUC**: ROC曲线下面积
5. **SNR Gain**: 信噪比增益
6. **Abstention Accuracy**: Abstain准确率
7. **Predictive Accuracy**: 预测准确率

### 4.2.2 响应级评估 (BeyondAIME)

#### 评估脚本
```bash
# 可选：外部模型（noapi）
export OPENAI_API_KEY=你的key
export OPENAI_API_BASE=https://noapi.ggb.today/v1
bash /root/SeedMitigating\ /rl/scripts/eval_frontier_noapi.sh

# 评估所有模型变体
bash /root/SeedMitigating\ /rl/scripts/eval_response_level_beyondaime.sh

# 生成Table 1
python /root/SeedMitigating\ /rl/evaluation/generate_table1.py

# 生成Figure 4
python /root/SeedMitigating\ /rl/evaluation/generate_figure4.py
```

#### 输出结果
- **Table 1**: 响应级置信度校准结果表（7项指标 × 所有模型）
- **Figure 4**: 校准图
  - 子图(a): 外部模型（noapi, gpt-4o-mini / gemini-2.5-flash）
  - 子图(b): 自研模型变体的置信度-准确率关系

#### AIME-2024/2025补充评估
```bash
# 评估AIME数据集
bash /root/SeedMitigating\ /rl/scripts/eval_response_level_aime.sh

# 生成Table 2
python /root/SeedMitigating\ /rl/evaluation/generate_table2.py
```

### 4.2.3 声明级评估 (BeyondAIME)

#### 声明正确性标注（noapi）
使用 noapi 便宜模型自动标注：
```bash
export OPENAI_API_KEY=你的key
export OPENAI_API_BASE=https://noapi.ggb.today/v1
export NOAPI_LABEL_MODEL=gpt-4o-mini   # 或 gemini-2.5-flash
```

#### 评估脚本
```bash
# 评估声明级校准（会先导出 claims_for_labeling.jsonl，再自动noapi标注）
bash /root/SeedMitigating\ /rl/scripts/eval_claim_level_beyondaime.sh

# 如已有标注文件/目录：
python /root/SeedMitigating\ /rl/evaluation/generate_table3.py --labels_file /path/to/claim_labels.jsonl
python /root/SeedMitigating\ /rl/evaluation/generate_figure5.py --labels_file /path/to/claim_labels.jsonl
```

## 4.3 行为校准目标验证

### 4.3.1 Adaptive Risk + Hallucination Reduction

**脚本**: `eval_adaptive_risk.sh`

评估不同风险阈值下的幻觉率和abstain率：
```bash
bash /root/SeedMitigating\ /rl/scripts/eval_adaptive_risk.sh

# 生成Figure 6 (9个子图)
python /root/SeedMitigating\ /rl/evaluation/generate_figure6.py
```

**验证点**:
- 自研模型：凹形abstain曲线（快速适应风险）
- 对比模型：凸形abstain曲线（风险适应缓慢）
- t=1时幻觉率降至0

### 4.3.2 Accuracy Preservation

**验证**: 在t=0（无abstain）时，自研模型准确率与Baseline一致

查看Table 1中的Pred Acc列即可验证。

### 4.3.3 Quantitative Calibration

**脚本**: `eval_quantitative_calibration.sh`

评估True Positive率和False Negative率：
```bash
bash /root/SeedMitigating\ /rl/scripts/eval_quantitative_calibration.sh

# 生成Figure 7 (4个子图)
python /root/SeedMitigating\ /rl/evaluation/generate_figure7.py
```

**验证点**:
- 自研模型：TP曲线在y=x上方，FN曲线在y=x下方
- 对比模型：TP曲线增长缓慢，FN曲线在低阈值时高于y=x

## 4.4 跨域迁移评估 (SimpleQA)

**脚本**: `eval_simpleqa.sh`

评估在SimpleQA数据集上的零样本迁移能力：
```bash
# 可选：外部模型（noapi）
export OPENAI_API_KEY=你的key
export OPENAI_API_BASE=https://noapi.ggb.today/v1
bash /root/SeedMitigating\ /rl/scripts/eval_frontier_noapi.sh

bash /root/SeedMitigating\ /rl/scripts/eval_simpleqa.sh

# 生成Table 4
python /root/SeedMitigating\ /rl/evaluation/generate_table4.py
```

**关键验证点**: Confidence-Brier变体的smECE、SNR Gain、Conf AUC应与外部模型（gpt-4o-mini / gemini-2.5-flash）相当
**评分口径**: SimpleQA使用LLM评分器（默认 `gpt-4o-mini`）判定 Correct / Incorrect / Not attempted，Not attempted 视为 abstain。请确保 `OPENAI_API_KEY` 可用。
**声明级外部模型**: 若要生成外部模型的声明级结果与标注，请设置：
```bash
export NOAPI_CLAIM_LEVEL=1
export NOAPI_LABEL_MODEL=gpt-4o-mini
bash /root/SeedMitigating\ /rl/scripts/eval_frontier_noapi.sh
```

## 4.5 测试时缩放 (Test-Time Scaling)

**脚本**: `eval_test_time_scaling.sh`

评估多轮预测聚合方法：
```bash
bash /root/SeedMitigating\ /rl/scripts/eval_test_time_scaling.sh

# 生成Figure 8 (3个子图)
python /root/SeedMitigating\ /rl/evaluation/generate_figure8.py
```

**对比方法**:
- 基准: Mean@k, Best@k, Majority@k
- 自研: Max Confidence, Confidence Weighted Majority

## 完整复现流程

### 步骤1: 训练所有模型变体 (约7-10天)
```bash
# 并行训练所有6个变体（如果有足够的NPU资源）
bash /root/SeedMitigating\ /rl/scripts/train_all_variants.sh
```

### 步骤2: 评估响应级校准 (约1-2天)
```bash
bash /root/SeedMitigating\ /rl/scripts/eval_all_response_level.sh
```

### 步骤3: 评估声明级校准 (约1-2天)
```bash
bash /root/SeedMitigating\ /rl/scripts/eval_all_claim_level.sh
```

### 步骤4: 行为校准验证 (约1天)
```bash
bash /root/SeedMitigating\ /rl/scripts/eval_all_behavioral_criteria.sh
```

### 步骤5: 跨域迁移评估 (约半天)
```bash
bash /root/SeedMitigating\ /rl/scripts/eval_simpleqa.sh
```

### 步骤6: 测试时缩放实验 (约半天)
```bash
bash /root/SeedMitigating\ /rl/scripts/eval_test_time_scaling.sh
```

### 步骤7: 生成所有表格和图表
```bash
bash /root/SeedMitigating\ /rl/scripts/generate_all_results.sh
```

## 输出结果

所有结果将保存在 `/data2/SeedMitigating-output/paper_reproduction/` 目录下：

```
output/paper_reproduction/
├── tables/
│   ├── table1_response_level_beyondaime.csv
│   ├── table2_response_level_aime.csv
│   ├── table3_claim_level_beyondaime.csv
│   └── table4_simpleqa.csv
├── figures/
│   ├── figure4_response_level_calibration.png
│   ├── figure5_claim_level_calibration.png
│   ├── figure6_adaptive_risk.png
│   ├── figure7_quantitative_calibration.png
│   └── figure8_test_time_scaling.png
├── checkpoints/
│   ├── baseline_ppo/
│   ├── confidence_brier/
│   ├── confidence_ce/
│   ├── ppo_value/
│   ├── confidence_prod/
│   └── confidence_min/
└── logs/
    └── [各种训练和评估日志]
```

## 注意事项

1. **NPU适配**: 所有脚本已针对NPU环境优化，使用 `trainer.device=npu`
2. **模型一致性**: 必须使用 Qwen3-4B-Instruct-2507 作为基础模型
3. **数据集版本**: 使用ByteDance-Seed 2025版本的BeyondAIME
4. **评估指标**: smECE实现参考Blasiok et al. [6] (ICLR 2024)
5. **外部模型**: 需要配置 noapi 的 `OPENAI_API_KEY` 与 `OPENAI_API_BASE`
6. **计算资源**: 建议使用8×NPU进行训练，单个变体训练约需1-2天

## 故障排查

### 常见问题

1. **NPU内存不足**: 减小 `gpu_memory_utilization` 或 `batch_size`
2. **训练不稳定**: 检查学习率和KL惩罚系数
3. **置信度提取失败**: 检查模型输出格式是否符合预期
4. **评估指标异常**: 确认数据集格式和标注正确性

### 日志查看
```bash
# 查看训练日志
tail -f /root/SeedMitigating\ /output/verl_logs/[experiment_name]/train.log

# 查看评估日志
tail -f /root/SeedMitigating\ /output/evaluation_logs/[eval_name].log
```

## 参考文献

- [6] Blasiok et al. "Smooth Calibration, Learnability, and Approximation." ICLR 2024.
- [31] Wei et al. "SimpleQA: A Simple Benchmark for Factual Question Answering." arXiv:2411.04368.
- [33] Yu et al. "DAPO: Direct Alignment via Preference Optimization." arXiv:2503.14476.

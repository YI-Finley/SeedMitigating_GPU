# 论文第4节实验复现 - 脚本清单

本文档列出了为复现论文第4节实验而创建的所有脚本和文档。

## 📁 文件结构

```
/root/SeedMitigating/rl/
├── scripts/                          # 训练和评估脚本
│   ├── train_baseline_ppo.sh        # 训练Baseline PPO
│   ├── train_confidence_brier.sh    # 训练Confidence-Brier
│   ├── train_confidence_ce.sh       # 训练Confidence-CE (GRPO)
│   ├── train_ppo_value.sh           # 训练PPO-Value
│   ├── train_confidence_prod.sh     # 训练Confidence-Prod
│   ├── train_confidence_min.sh      # 训练Confidence-Min
│   ├── train_all_variants.sh        # 训练所有6个变体（主脚本）
│   └── eval_response_level_beyondaime.sh  # 响应级评估
├── evaluation/                       # 评估和结果生成
│   ├── generate_table1.py           # 生成Table 1
│   └── (更多评估脚本待创建)
├── QUICK_START.md                    # 快速开始指南（推荐阅读）
├── REPRODUCTION_GUIDE.md             # 完整复现指南
├── PAPER_EXPERIMENTS.md              # 论文实验详细清单
├── verl_custom_reward.py             # 自定义奖励函数
└── rewards.py                        # 奖励函数实现
```

## 📋 脚本清单

### 1. 训练脚本（6个模型变体）

#### 1.1 Baseline PPO
**文件**: `scripts/train_baseline_ppo.sh`
**描述**: 标准PPO训练，使用二元奖励（+1正确，-1错误）
**对应论文**: Section 4.1 - Baseline模型
**运行**:
```bash
bash /root/SeedMitigating\ /rl/scripts/train_baseline_ppo.sh
```

#### 1.2 Confidence-Brier
**文件**: `scripts/train_confidence_brier.sh`
**描述**: Verbalized Confidence + Brier Score（均匀分布风险阈值）
**对应论文**: Section 3.2.2 - Uniform Distribution
**奖励公式**: R = 2p·valid(y) - p²
**运行**:
```bash
bash /root/SeedMitigating\ /rl/scripts/train_confidence_brier.sh
```

#### 1.3 Confidence-CE
**文件**: `scripts/train_confidence_ce.sh`
**描述**: Verbalized Confidence + Cross-Entropy（Beta分布，使用GRPO）
**对应论文**: Section 3.2.2 - Beta Distribution
**奖励公式**: R = [log((1-ε)/ε)]^(-1) * [valid(y)·log(p'/ε) + ...]
**特点**: 使用GRPO算法提升训练稳定性
**运行**:
```bash
bash /root/SeedMitigating\ /rl/scripts/train_confidence_ce.sh
```

#### 1.4 PPO-Value
**文件**: `scripts/train_ppo_value.sh`
**描述**: PPO Critic网络输出作为置信度（无额外置信度token）
**对应论文**: Section 3.2.3 - Critic Value-based Confidence
**奖励公式**: R = 2V(s)·valid(y) - V(s)²
**运行**:
```bash
bash /root/SeedMitigating\ /rl/scripts/train_ppo_value.sh
```

#### 1.5 Confidence-Prod
**文件**: `scripts/train_confidence_prod.sh`
**描述**: 声明级置信度 + 乘积聚合
**对应论文**: Section 3.3 - Claim-level Calibration (Product)
**奖励公式**: R = 2(∏p_i)·valid(y) - (∏p_i)²
**运行**:
```bash
bash /root/SeedMitigating\ /rl/scripts/train_confidence_prod.sh
```

#### 1.6 Confidence-Min
**文件**: `scripts/train_confidence_min.sh`
**描述**: 声明级置信度 + 最小值聚合
**对应论文**: Section 3.3 - Claim-level Calibration (Minimum)
**奖励公式**: R = 2(min p_i)·valid(y) - (min p_i)²
**运行**:
```bash
bash /root/SeedMitigating\ /rl/scripts/train_confidence_min.sh
```

#### 1.7 训练所有变体（主脚本）
**文件**: `scripts/train_all_variants.sh`
**描述**: 一键训练所有6个模型变体（支持串行/并行）
**运行**:
```bash
bash /root/SeedMitigating\ /rl/scripts/train_all_variants.sh
# 选择模式：1=串行（推荐），2=并行（需48 NPU）
```

### 2. 评估脚本

#### 2.1 响应级评估（BeyondAIME）
**文件**: `scripts/eval_response_level_beyondaime.sh`
**描述**: 评估所有模型在BeyondAIME数据集上的响应级置信度校准
**对应论文**: Section 4.2.2
**运行**:
```bash
bash /root/SeedMitigating\ /rl/scripts/eval_response_level_beyondaime.sh
```

#### 2.2 生成Table 1
**文件**: `evaluation/generate_table1.py`
**描述**: 生成响应级置信度校准结果表（7项指标）
**输出**: `output/paper_reproduction/tables/table1_response_level_beyondaime.csv`
**运行**:
```bash
python /root/SeedMitigating\ /rl/evaluation/generate_table1.py
```

### 3. 核心代码模块

#### 3.1 自定义奖励函数
**文件**: `verl_custom_reward.py`
**描述**: VERL框架的自定义奖励函数接口，实现6种校准策略
**函数**:
- `compute_score_baseline()`: Baseline二元奖励
- `compute_score_verbalized_brier()`: Brier Score奖励
- `compute_score_verbalized_ce()`: Cross-Entropy奖励
- `compute_score_critic_value()`: Critic Value奖励
- `compute_score_claim_product()`: 声明级乘积聚合
- `compute_score_claim_minimum()`: 声明级最小值聚合

#### 3.2 奖励函数实现
**文件**: `rewards.py`
**描述**: 所有奖励函数的具体实现
**主要函数**:
- `extract_answer()`: 从输出中提取答案
- `extract_confidence()`: 从输出中提取置信度
- `is_correct()`: 判断答案是否正确
- `binary_reward()`: 二元奖励
- `verbalized_confidence_brier_reward()`: Brier Score奖励
- `verbalized_confidence_ce_reward()`: Cross-Entropy奖励
- `critic_value_reward()`: Critic Value奖励
- `claim_level_confidence_reward()`: 声明级奖励

### 4. 评估基础设施（已有代码）

#### 4.1 置信度校准指标
**文件**: `/root/SeedMitigating/eval/confidence_calibration_metrics.py`
**描述**: 实现论文Section 4.2的7个定量评估指标
**类**: `ConfidenceCalibrationMetrics`
**方法**:
- `compute_smoothed_ece()`: Smoothed ECE
- `compute_brier_score()`: Brier Score
- `compute_nll()`: Negative Log-Likelihood
- `compute_confidence_auc()`: Confidence AUC
- `compute_snr_gain()`: SNR Gain
- `compute_abstention_accuracy()`: Abstention Accuracy
- `compute_predictive_accuracy()`: Predictive Accuracy

#### 4.2 行为校准指标
**文件**: `/root/SeedMitigating/eval/behavioral_calibration_metrics.py`
**描述**: 实现论文Section 4.3的4个校准目标
**类**: `BehavioralCalibrationMetrics`
**方法**:
- `compute_adaptive_risk()`: Adaptive Risk
- `compute_accuracy_preservation()`: Accuracy Preservation
- `compute_hallucination_reduction()`: Hallucination Reduction
- `compute_quantitative_calibration()`: Quantitative Calibration

#### 4.3 可视化模块
**文件**: `/root/SeedMitigating/eval/visualize.py`
**描述**: 生成论文同款图表
**函数**:
- `visualize_adaptive_risk()`: Figure 6同款凹形曲线
- `visualize_hallucination_reduction()`: SNR/SNR Gain曲线
- `visualize_quantitative_calibration()`: Figure 7同款TP/FN曲线
- `visualize_all_metrics()`: 生成所有图表

#### 4.4 评估脚本
**文件**: `/root/SeedMitigating/scripts/evaluate_model_fixed.py`
**描述**: 主评估脚本，支持BeyondAIME、AIME、SimpleQA数据集
**功能**:
- 加载训练好的模型
- 生成模型输出
- 提取答案和置信度
- 计算所有评估指标
- 保存评估结果

### 5. 文档

#### 5.1 快速开始指南（推荐）
**文件**: `QUICK_START.md`
**描述**: 简化的复现指南，包含完整的训练和评估流程
**内容**:
- 环境配置
- 快速开始命令
- 详细步骤说明
- 输出结果说明
- 故障排查

#### 5.2 完整复现指南
**文件**: `REPRODUCTION_GUIDE.md`
**描述**: 详细的实验复现指南，包含所有章节
**内容**:
- 第4节所有实验的详细说明
- 训练配置参数
- 评估指标定义
- 结果呈现要求

#### 5.3 论文实验清单
**文件**: `PAPER_EXPERIMENTS.md`
**描述**: 论文第4节实验的详细清单（用户提供）
**内容**:
- 4.1 训练配置
- 4.2 置信度校准定量评估
- 4.3 行为校准目标验证
- 4.4 跨域迁移评估
- 4.5 测试时缩放

## 🚀 使用流程

### 完整流程（推荐）

```bash
# 1. 阅读快速开始指南
cat /root/SeedMitigating\ /rl/QUICK_START.md

# 2. 训练所有模型变体（约7-10天）
bash /root/SeedMitigating\ /rl/scripts/train_all_variants.sh

# 3. 评估所有模型（约2-3天）
bash /root/SeedMitigating\ /rl/scripts/eval_response_level_beyondaime.sh

# 4. 生成表格和图表
python /root/SeedMitigating\ /rl/evaluation/generate_table1.py
```

### 单个模型流程

```bash
# 1. 训练单个模型（例如Confidence-Brier）
bash /root/SeedMitigating\ /rl/scripts/train_confidence_brier.sh

# 2. 评估该模型
python /root/SeedMitigating\ /scripts/evaluate_model_fixed.py \
    --model_path /root/SeedMitigating\ /output/paper_reproduction/checkpoints/confidence_brier \
    --base_model Qwen/Qwen3-4B-Instruct-2507 \
    --dataset beyondaime \
    --output_dir /root/SeedMitigating\ /output/evaluation/confidence_brier \
    --device npu:0

# 3. 查看结果
cat /root/SeedMitigating\ /output/evaluation/confidence_brier/results.json
```

## 📊 输出结果

所有结果保存在 `/data2/SeedMitigating-output/paper_reproduction/` 目录：

```
output/paper_reproduction/
├── checkpoints/              # 训练好的模型（约30GB/模型）
│   ├── baseline_ppo/
│   ├── confidence_brier/
│   ├── confidence_ce/
│   ├── ppo_value/
│   ├── confidence_prod/
│   └── confidence_min/
├── evaluation/               # 评估结果（JSON格式）
│   └── response_level/
│       └── beyondaime/
│           ├── baseline_ppo/results.json
│           ├── confidence_brier/results.json
│           └── ...
├── tables/                   # 论文表格（CSV格式）
│   └── table1_response_level_beyondaime.csv
└── figures/                  # 论文图表（PDF格式）
    └── (待生成)
```

## ⚙️ 配置说明

### 训练配置（所有模型统一）

```yaml
# 模型配置
基础模型: Qwen3-4B-Instruct-2507
训练数据: DAPO-Math-17k (17k数学问题)

# 训练参数
训练轮数: 30 epochs
学习率: 1e-5 (Actor和Critic)
Batch Size: 128 (全局)
Mini Batch Size: 32
Micro Batch Size: 4 (per GPU)

# RL参数
算法: PPO (或GRPO for Confidence-CE)
γ (gamma): 0.99
λ (lambda): 0.95
Advantage Estimator: GAE (或GRPO)

# 生成参数
温度: 0.7
Top-p: 0.9
Max Prompt Length: 512
Max Response Length: 2048

# 硬件配置
设备: NPU
NPU数量: 8 per node
节点数: 1
GPU Memory Utilization: 0.4
```

### 评估配置

```yaml
# 数据集
BeyondAIME: 100道超难数学题
AIME-2024: 30道AIME 2024题目
AIME-2025: 30道AIME 2025题目
SimpleQA: 长尾事实性问答

# 评估参数
Batch Size: 1
Risk Threshold: 0.5 (用于Abstention)
Device: npu:0
```

## 🔧 故障排查

### 常见问题

1. **NPU内存不足**
   - 减小 `gpu_memory_utilization` (0.4 → 0.3)
   - 减小 `batch_size` (128 → 64)

2. **训练不稳定**
   - 降低学习率 (1e-5 → 5e-6)
   - 检查KL惩罚系数

3. **置信度提取失败**
   - 检查模型输出格式
   - 确保包含 "Confidence: 0.XX"

4. **评估指标异常**
   - 检查数据集格式
   - 确认ground_truth字段存在

### 日志位置

```bash
# 训练日志
/data2/SeedMitigating-output/paper_reproduction/checkpoints/[model]/train.log

# 评估日志
/data2/SeedMitigating-output/paper_reproduction/evaluation/[eval]/eval.log

# WandB日志（如果启用）
https://wandb.ai/your-project/behavioral_calibration_paper
```

## 📝 脚本清单（已更新）

### 评估脚本
- [x] `scripts/eval_response_level_aime.sh` - AIME评估
- [x] `scripts/eval_claim_level_beyondaime.sh` - 声明级评估
- [x] `scripts/eval_adaptive_risk.sh` - Adaptive Risk评估
- [x] `scripts/eval_quantitative_calibration.sh` - Quantitative Calibration评估
- [x] `scripts/eval_simpleqa.sh` - SimpleQA评估
- [x] `scripts/eval_test_time_scaling.sh` - 测试时缩放评估
- [x] `scripts/eval_frontier_noapi.sh` - 外部模型(noapi)评估（可选 `NOAPI_CLAIM_LEVEL=1` 生成声明级结果）
- [x] `scripts/generate_all_results.sh` - 生成所有结果（主脚本）
- [x] `scripts/evaluate_external_model.py` - 外部模型评估（noapi）
- [x] `scripts/label_claims_noapi.py` - claim 标注（noapi）

### 结果生成脚本
- [x] `evaluation/generate_table2.py` - 生成Table 2 (AIME)
- [x] `evaluation/generate_table3.py` - 生成Table 3 (声明级)
- [x] `evaluation/generate_table4.py` - 生成Table 4 (SimpleQA)
- [x] `evaluation/generate_figure4.py` - 生成Figure 4 (校准图)
- [x] `evaluation/generate_figure5.py` - 生成Figure 5 (声明级校准图)
- [x] `evaluation/generate_figure6.py` - 生成Figure 6 (Adaptive Risk)
- [x] `evaluation/generate_figure7.py` - 生成Figure 7 (Quantitative Calibration)
- [x] `evaluation/generate_figure8.py` - 生成Figure 8 (测试时缩放)

## 📚 参考资料

- **论文**: Mitigating LLM Hallucination via Behaviorally Calibrated Reinforcement Learning
- **VERL框架**: `/root/SeedMitigating/verl/`
- **评估代码**: `/root/SeedMitigating/eval/`
- **数据集**: `/root/SeedMitigating/data/`

## ✅ 已完成的工作

- [x] 6个训练脚本（所有模型变体）
- [x] 训练主脚本（train_all_variants.sh）
- [x] 自定义奖励函数（verl_custom_reward.py）
- [x] 奖励函数实现（rewards.py）
- [x] 响应级评估脚本（eval_response_level_beyondaime.sh）
- [x] Table 1生成脚本（generate_table1.py）
- [x] 快速开始指南（QUICK_START.md）
- [x] 完整复现指南（REPRODUCTION_GUIDE.md）
- [x] 脚本清单文档（本文档）

## 🎯 下一步

1. 运行训练脚本，训练所有6个模型变体
2. 创建剩余的评估脚本
3. 创建结果生成脚本
4. 运行完整的评估流程
5. 生成所有表格和图表
6. 验证结果与论文一致性

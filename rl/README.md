# 论文第4节实验完整复现

本项目提供《Mitigating LLM Hallucination via Behaviorally Calibrated Reinforcement Learning》论文第4节（Experiments）的完整复现方案。

## 🎯 复现目标

复现论文第4节的所有实验，包括：
- **4.1** 训练6个模型变体（Baseline + 5种校准策略）
- **4.2** 置信度校准定量评估（7项指标）
- **4.3** 行为校准目标验证（4个校准目标）
- **4.4** 跨域迁移评估（SimpleQA）
- **4.5** 测试时缩放实验

## 📚 文档导航

| 文档 | 描述 | 推荐阅读顺序 |
|------|------|------------|
| **[QUICK_START.md](QUICK_START.md)** | 快速开始指南，包含完整流程 | ⭐ 首先阅读 |
| **[SCRIPTS_INDEX.md](SCRIPTS_INDEX.md)** | 所有脚本的详细清单和说明 | ⭐ 其次阅读 |
| **[PAPER_EXPERIMENTS.md](PAPER_EXPERIMENTS.md)** | 论文实验的详细清单 | 参考 |
| **[REPRODUCTION_GUIDE.md](REPRODUCTION_GUIDE.md)** | 完整的复现指南 | 参考 |

## 🚀 快速开始

### 1. 训练所有模型（约7-10天）

```bash
# 串行训练所有6个模型变体
bash /root/SeedMitigating\ /rl/scripts/train_all_variants.sh
```

### 2. 评估所有模型（约2-3天）

```bash
# 响应级评估（BeyondAIME）
bash /root/SeedMitigating\ /rl/scripts/eval_response_level_beyondaime.sh
```

### 3. 生成结果

```bash
# 生成Table 1
python /root/SeedMitigating\ /rl/evaluation/generate_table1.py
```

## 📁 项目结构

```
/root/SeedMitigating/rl/
├── scripts/                    # 训练和评估脚本
│   ├── train_*.sh             # 6个训练脚本
│   ├── train_all_variants.sh  # 训练主脚本
│   └── eval_*.sh              # 评估脚本
├── evaluation/                 # 结果生成脚本
│   └── generate_*.py          # 生成表格和图表
├── verl_custom_reward.py       # 自定义奖励函数
├── rewards.py                  # 奖励函数实现
├── QUICK_START.md              # 快速开始指南 ⭐
├── SCRIPTS_INDEX.md            # 脚本清单 ⭐
├── PAPER_EXPERIMENTS.md        # 论文实验清单
└── REPRODUCTION_GUIDE.md       # 完整复现指南
```

## 🔬 6个模型变体

| 模型 | 描述 | 训练脚本 |
|------|------|---------|
| **Baseline PPO** | 标准PPO，二元奖励 | `train_baseline_ppo.sh` |
| **Confidence-Brier** | Verbalized Confidence + Brier Score | `train_confidence_brier.sh` |
| **Confidence-CE** | Verbalized Confidence + Cross-Entropy (GRPO) | `train_confidence_ce.sh` |
| **PPO-Value** | Critic Value作为置信度 | `train_ppo_value.sh` |
| **Confidence-Prod** | 声明级置信度 + 乘积聚合 | `train_confidence_prod.sh` |
| **Confidence-Min** | 声明级置信度 + 最小值聚合 | `train_confidence_min.sh` |

## 📊 评估指标

### 置信度校准指标（Section 4.2）
- **smECE**: Smoothed ECE（越小越好）
- **Brier**: Brier Score（越小越好）
- **NLL**: Negative Log-Likelihood（越小越好）
- **Conf AUC**: Confidence AUC（越大越好）
- **SNR Gain**: 信噪比增益（越大越好）
- **Abs Acc**: Abstention Accuracy（越大越好）
- **Pred Acc**: Predictive Accuracy（参考）

### 行为校准目标（Section 4.3）
1. **Adaptive Risk**: 自适应风险调整
2. **Accuracy Preservation**: 准确率保留
3. **Hallucination Reduction**: 幻觉减少
4. **Quantitative Calibration**: 定量校准

## 🛠️ 环境配置

- **虚拟环境**: `truthrl-verl-npu`
- **基础模型**: Qwen3-4B-Instruct-2507
- **训练框架**: VERL (NPU适配版本)
- **硬件**: 8×NPU per node
- **数据集**: DAPO-Math-17k, BeyondAIME, AIME, SimpleQA

## 📈 预期结果

### Table 1 (BeyondAIME响应级)
- **Confidence-Brier/CE**: 最佳的smECE、Brier、NLL
- **PPO-Value**: 较好的Conf AUC和SNR Gain
- **Baseline PPO**: 最差的校准指标

### Figure 6 (Adaptive Risk)
- **自研模型**: 凹形abstain曲线，幻觉率快速下降
- **Baseline**: 凸形abstain曲线，幻觉率下降缓慢

### Figure 7 (Quantitative Calibration)
- **自研模型**: TP曲线在y=x上方，FN曲线在y=x下方
- **Baseline**: TP曲线低于y=x，FN曲线高于y=x

## 🔧 故障排查

### NPU内存不足
```bash
# 减小gpu_memory_utilization或batch_size
actor_rollout_ref.rollout.gpu_memory_utilization=0.3
data.train_batch_size=64
```

### 训练不稳定
```bash
# 降低学习率
actor_rollout_ref.actor.optim.lr=5e-6
```

### 查看日志
```bash
# 训练日志
tail -f /data2/SeedMitigating-output/paper_reproduction/checkpoints/[model]/train.log

# 评估日志
tail -f /data2/SeedMitigating-output/paper_reproduction/evaluation/[eval]/eval.log
```

## 📝 已完成的工作

- ✅ 6个训练脚本（所有模型变体）
- ✅ 训练主脚本（train_all_variants.sh）
- ✅ 自定义奖励函数（verl_custom_reward.py）
- ✅ 奖励函数实现（rewards.py）
- ✅ 响应级评估脚本（eval_response_level_beyondaime.sh）
- ✅ Table 1生成脚本（generate_table1.py）
- ✅ 完整文档（4个MD文件）

## 🎯 下一步

1. **运行训练**: 使用 `train_all_variants.sh` 训练所有模型
2. **运行评估**: 使用 `eval_response_level_beyondaime.sh` 评估模型
3. **生成结果**: 使用 `generate_table1.py` 生成表格
4. **创建剩余脚本**: 根据 `SCRIPTS_INDEX.md` 中的待办清单

## 📖 参考资料

- **论文**: Mitigating LLM Hallucination via Behaviorally Calibrated Reinforcement Learning
- **VERL框架**: `/root/SeedMitigating/verl/`
- **评估代码**: `/root/SeedMitigating/eval/`
  - `confidence_calibration_metrics.py`: 置信度校准指标
  - `behavioral_calibration_metrics.py`: 行为校准指标
  - `visualize.py`: 可视化模块
- **数据集**: `/root/SeedMitigating/data/`

## 💡 使用建议

1. **首先阅读** [QUICK_START.md](QUICK_START.md) 了解完整流程
2. **查看** [SCRIPTS_INDEX.md](SCRIPTS_INDEX.md) 了解所有脚本
3. **参考** [PAPER_EXPERIMENTS.md](PAPER_EXPERIMENTS.md) 了解论文实验细节
4. **遇到问题** 查看 [QUICK_START.md](QUICK_START.md) 的故障排查部分

## 📧 联系方式

如有问题，请查看：
- 训练日志: `/data2/SeedMitigating-output/paper_reproduction/checkpoints/[model]/train.log`
- 评估日志: `/data2/SeedMitigating-output/paper_reproduction/evaluation/[eval]/eval.log`
- WandB: https://wandb.ai (如果启用)

---

**注意**: 本复现专注于6个自研模型变体，暂不包含外部前沿模型（GPT-5、Claude等）的对比评估。

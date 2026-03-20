# SeedMitigating GPU/SLURM 版本

这是原始 `SeedMitigating` 项目的一个 GPU 化改写版本，目标是让它能够在 CUDA GPU 和 SLURM 集群环境下运行，尤其适合你学校的 A100 节点工作流。

建议优先使用新增的 [gpu](gpu) 启动脚本和 [slurm](slurm) 里的 `sbatch` 模板。原始 NPU 说明保存在 [README_NPU_ORIGINAL.md](README_NPU_ORIGINAL.md)。

## 这个版本做了什么

- 新增了 GPU 版训练和评估入口，放在 [gpu](gpu)
- 新增了适合 A100 集群的 SLURM 模板，放在 [slurm](slurm)
- 主评估脚本已经支持 `cuda:*` 和 `npu:*`
- 增加了 GPU 环境说明：[ENV_SETUP_GPU.md](ENV_SETUP_GPU.md)
- 增加了集群使用说明：[GPU_CLUSTER_GUIDE.md](GPU_CLUSTER_GUIDE.md)

## 目录说明

- [gpu](gpu)：GPU 优先的训练/评估封装脚本
- [slurm](slurm)：可直接改后提交的 `sbatch` 模板
- [rl](rl)：论文核心训练脚本和图表生成逻辑
- [scripts](scripts)：模型评估主入口
- [data](data)：论文复现用到的数据
- [ENV_SETUP_GPU.md](ENV_SETUP_GPU.md)：环境配置说明
- [GPU_CLUSTER_GUIDE.md](GPU_CLUSTER_GUIDE.md)：集群运行说明

## 最短上手流程

如果你只想按最短路径跑起来，建议按这个顺序：

1. 把项目放到集群上：
   `/home/comp/23481501/datasets/SeedMitigating_GPU`
2. 按 [ENV_SETUP_GPU.md](ENV_SETUP_GPU.md) 创建独立 Python 环境
3. 先验证环境是否正常
4. 再用 [slurm](slurm) 里的模板提交作业

## 一步一步操作

### 1. 把项目上传到集群

推荐放在：

```bash
/home/comp/23481501/datasets/SeedMitigating_GPU
```

上传后先检查目录是否完整：

```bash
ls /home/comp/23481501/datasets/SeedMitigating_GPU
```

你应该能看到：

- `gpu`
- `slurm`
- `rl`
- `scripts`
- `data`
- `verl`

### 2. 创建干净的 Python 环境

先看 [ENV_SETUP_GPU.md](ENV_SETUP_GPU.md)。核心原则是：

- 不要继续依赖现在混合的 `base + ~/.local` 环境
- 给这个项目单独建一个环境
- 运行时设置 `PYTHONNOUSERSITE=1`

推荐方式：

```bash
module load cuda/11.4
python3.10 -m venv /home/comp/23481501/venvs/seedmitigating-gpu
source /home/comp/23481501/venvs/seedmitigating-gpu/bin/activate
export PYTHONNOUSERSITE=1
```

然后安装依赖：

```bash
python -m pip install --upgrade pip setuptools wheel
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
cd /home/comp/23481501/datasets/SeedMitigating_GPU
pip install -r requirements-gpu.txt
pip install -e ./verl
```

### 3. 提交作业前先验证环境

建议先跑这几条：

```bash
source /home/comp/23481501/venvs/seedmitigating-gpu/bin/activate
export PYTHONNOUSERSITE=1

python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.device_count())"
python -c "import transformers, ray; print(transformers.__version__); print(ray.__version__)"
python -c "import vllm; print(vllm.__version__)"
```

如果这里 `import torch` 都过不了，就不要先提交训练任务，先把环境修好。

### 4. 选择要运行的内容

推荐的直接入口：

- 全量训练：`bash gpu/run_full_training.sh`
- 全量评估：`bash gpu/run_evaluation.sh`
- Test-Time Scaling：`bash gpu/run_test_time_scaling.sh`

推荐直接使用的 `sbatch` 模板：

- [train_all_variants_a100_2gpu.sbatch](slurm/train_all_variants_a100_2gpu.sbatch)
- [evaluate_all_a100_1gpu.sbatch](slurm/evaluate_all_a100_1gpu.sbatch)
- [test_time_scaling_a100_1gpu.sbatch](slurm/test_time_scaling_a100_1gpu.sbatch)

### 5. 通过 SLURM 提交

例如提交训练：

```bash
cd /home/comp/23481501/datasets/SeedMitigating_GPU
sbatch slurm/train_all_variants_a100_2gpu.sbatch
```

查看任务状态：

```bash
squeue -u $USER
```

### 6. 结果输出在哪里

默认输出目录是：

```bash
/home/comp/23481501/datasets/SeedMitigating_GPU/output_gpu/paper_reproduction
```

主要包括：

- `checkpoints/`
- `evaluation/`
- `test_time_scaling/`

## 常见工作流

### 全量训练

```bash
sbatch slurm/train_all_variants_a100_2gpu.sbatch
```

### 全量评估

```bash
sbatch slurm/evaluate_all_a100_1gpu.sbatch
```

### Test-Time Scaling

```bash
sbatch slurm/test_time_scaling_a100_1gpu.sbatch
```

## 这个 GPU 版的注意事项

- 原来的 `rl/scripts/*.sh` 和顶层 NPU 脚本仍然保留，主要用于参考
- 实际建议使用新的 [gpu](gpu) 封装脚本，而不是旧的 NPU 脚本
- 新的 GPU 入口默认假设你使用独立虚拟环境，并设置了 `PYTHONNOUSERSITE=1`
- 主评估 Python 脚本已经不再写死为 NPU

## 上传到 GitHub

如果你想把这个项目上传到 GitHub，最简单的流程是：

```bash
cd /path/to/SeedMitigating_GPU
git init
git add .
git commit -m "Initial GPU/SLURM port of SeedMitigating"
git branch -M main
git remote add origin git@github.com:YI-Finley/SeedMitigating_GPU.git
git push -u origin main
```

如果你已经在 GitHub 上创建好了仓库，但远程有初始 README，那么建议先：

```bash
git pull --rebase origin main
git push -u origin main
```

如果 rebase 时有冲突，再手动处理。

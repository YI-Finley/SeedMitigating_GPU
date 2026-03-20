# SeedMitigating GPU/SLURM Port

This repository is a GPU/SLURM-ready port of the original `SeedMitigating` project. It is prepared for CUDA GPUs, A100 nodes, and your school cluster workflow.

Use the new launchers under [gpu](gpu) and the ready-made SLURM templates under [slurm](slurm). The original NPU-oriented notes were preserved in [README_NPU_ORIGINAL.md](README_NPU_ORIGINAL.md).

## What This Version Changes

- Adds GPU-safe training and evaluation launchers under [gpu](gpu)
- Adds SLURM templates for your A100 cluster under [slurm](slurm)
- Makes the main evaluation Python entrypoints accept both `cuda:*` and `npu:*`
- Adds cluster-facing setup notes in [ENV_SETUP_GPU.md](ENV_SETUP_GPU.md)
- Keeps the original project structure so the paper scripts and data are still easy to map

## Repository Layout

- [gpu](gpu): new GPU-first train/eval wrappers
- [slurm](slurm): `sbatch` templates
- [rl](rl): core paper training scripts and evaluation generators
- [scripts](scripts): model evaluation entrypoints
- [data](data): datasets used by the paper reproduction flow
- [ENV_SETUP_GPU.md](ENV_SETUP_GPU.md): environment setup notes
- [GPU_CLUSTER_GUIDE.md](GPU_CLUSTER_GUIDE.md): cluster-specific notes

## Quick Start

If you only want the shortest working path, do this:

1. Upload this repo to your cluster at:
   `/home/comp/23481501/datasets/SeedMitigating_GPU`
2. Create the dedicated virtualenv described in [ENV_SETUP_GPU.md](ENV_SETUP_GPU.md)
3. Verify the environment:
   ```bash
   source /home/comp/23481501/venvs/seedmitigating-gpu/bin/activate
   export PYTHONNOUSERSITE=1
   python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
   ```
4. Submit one of the provided SLURM jobs:
   ```bash
   cd /home/comp/23481501/datasets/SeedMitigating_GPU
   sbatch slurm/train_all_variants_a100_2gpu.sbatch
   ```

## Step-by-Step Usage

### 1. Put the Project on the Cluster

Recommended location:

```bash
/home/comp/23481501/datasets/SeedMitigating_GPU
```

Check that the copy is complete:

```bash
ls /home/comp/23481501/datasets/SeedMitigating_GPU
```

You should see directories like `gpu`, `slurm`, `rl`, `scripts`, `data`, and `verl`.

### 2. Create a Clean Python Environment

Read [ENV_SETUP_GPU.md](ENV_SETUP_GPU.md) first. The important point is: do not rely on your current mixed `base + ~/.local` Python setup.

Recommended pattern:

```bash
module load cuda/11.4
python3.10 -m venv /home/comp/23481501/venvs/seedmitigating-gpu
source /home/comp/23481501/venvs/seedmitigating-gpu/bin/activate
export PYTHONNOUSERSITE=1
```

Then install the dependencies:

```bash
python -m pip install --upgrade pip setuptools wheel
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
cd /home/comp/23481501/datasets/SeedMitigating_GPU
pip install -r requirements-gpu.txt
pip install -e ./verl
```

### 3. Verify the Environment Before Running Jobs

Run these checks from the clean environment:

```bash
source /home/comp/23481501/venvs/seedmitigating-gpu/bin/activate
export PYTHONNOUSERSITE=1

python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.device_count())"
python -c "import transformers, ray; print(transformers.__version__); print(ray.__version__)"
python -c "import vllm; print(vllm.__version__)"
```

If `import torch` fails here, fix the environment first before submitting any training job.

### 4. Choose What You Want to Run

Recommended entrypoints:

- Full training: `bash gpu/run_full_training.sh`
- Full evaluation: `bash gpu/run_evaluation.sh`
- Test-time scaling: `bash gpu/run_test_time_scaling.sh`

Recommended SLURM templates:

- [train_all_variants_a100_2gpu.sbatch](slurm/train_all_variants_a100_2gpu.sbatch)
- [evaluate_all_a100_1gpu.sbatch](slurm/evaluate_all_a100_1gpu.sbatch)
- [test_time_scaling_a100_1gpu.sbatch](slurm/test_time_scaling_a100_1gpu.sbatch)

### 5. Submit Through SLURM

Example:

```bash
cd /home/comp/23481501/datasets/SeedMitigating_GPU
sbatch slurm/train_all_variants_a100_2gpu.sbatch
```

Check job status:

```bash
squeue -u $USER
```

### 6. Where Results Go

By default, outputs are written under:

```bash
/home/comp/23481501/datasets/SeedMitigating_GPU/output_gpu/paper_reproduction
```

That includes:

- `checkpoints/`
- `evaluation/`
- `test_time_scaling/`

## Common Workflows

### Full Training

```bash
sbatch slurm/train_all_variants_a100_2gpu.sbatch
```

### Full Evaluation

```bash
sbatch slurm/evaluate_all_a100_1gpu.sbatch
```

### Test-Time Scaling

```bash
sbatch slurm/test_time_scaling_a100_1gpu.sbatch
```

## Notes About This GPU Port

- The legacy `rl/scripts/*.sh` and older top-level NPU scripts are still present for reference.
- The actual recommended path is to use the new [gpu](gpu) wrappers, not the old NPU shell scripts.
- The GPU launchers assume a dedicated virtualenv and set `PYTHONNOUSERSITE=1`.
- The main Python evaluation entrypoints were made device-flexible, so they are no longer hard-wired to NPU.

## GitHub Upload Notes

If you want to upload this project to GitHub later, the simplest workflow is:

```bash
cd /path/to/SeedMitigating_GPU
git init
git add .
git commit -m "Initial GPU/SLURM port"
git branch -M main
git remote add origin git@github.com:YI-Finley/SeedMitigating_GPU.git
git push -u origin main
```

If you create the repository on GitHub first, update the remote URL above if you choose a different repo name.

# GPU Cluster Guide

This directory was prepared for your school SLURM cluster.

## Assumed Cluster Defaults

- Partition: `short`
- GPU type: `A100`
- Common training target: `2` GPUs
- Common evaluation target: `1` GPU
- CUDA module: `cuda/11.4`
- Virtualenv activation: `/home/comp/23481501/venvs/seedmitigating-gpu/bin/activate`

## Main GPU Scripts

- [gpu/run_full_training.sh](gpu/run_full_training.sh)
- [gpu/run_evaluation.sh](gpu/run_evaluation.sh)
- [gpu/run_test_time_scaling.sh](gpu/run_test_time_scaling.sh)

## Important Environment Variables

- `SEEDMIT_PROJECT_ROOT`
- `SEEDMIT_OUTPUT_BASE`
- `SEEDMIT_HF_CACHE`
- `SEEDMIT_ENV_ACTIVATE`
- `SEEDMIT_NUM_GPUS`
- `SEEDMIT_CUDA_MODULE`
- `SEEDMIT_BASE_MODEL`

All GPU launchers source [gpu/common.sh](gpu/common.sh), which supplies defaults for those variables.

## Typical Workflow

1. Create and verify the dedicated virtualenv
2. Copy this repo to `/home/comp/23481501/datasets/SeedMitigating_GPU`
3. Adjust the sbatch template if you want different output paths or walltime
4. Submit training or evaluation through `sbatch`

## Default Output Layout

- Checkpoints: `$SEEDMIT_OUTPUT_BASE/checkpoints`
- Evaluation: `$SEEDMIT_OUTPUT_BASE/evaluation`
- Test-time scaling: `$SEEDMIT_OUTPUT_BASE/test_time_scaling`

## Safety Defaults

- `PYTHONNOUSERSITE=1`
- `WANDB_MODE=offline`
- GPU detection from `CUDA_VISIBLE_DEVICES` or SLURM GPU metadata
- Smaller default batch settings than the original 8-NPU scripts

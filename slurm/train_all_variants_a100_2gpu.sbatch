#!/bin/bash
#SBATCH -J seedmit-train
#SBATCH -p short
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:a100:2
#SBATCH --mem=300G
#SBATCH -c 16
#SBATCH -t 240:00:00
#SBATCH -D /home/comp/23481501/datasets/SeedMitigating_GPU
#SBATCH -o /home/comp/23481501/datasets/SeedMitigating_GPU/output/slurm-train-%j.out
#SBATCH -e /home/comp/23481501/datasets/SeedMitigating_GPU/output/slurm-train-%j.err

export SEEDMIT_PROJECT_ROOT=/home/comp/23481501/datasets/SeedMitigating_GPU
export SEEDMIT_ENV_ACTIVATE=/home/comp/23481501/venvs/seedmitigating-gpu/bin/activate
export SEEDMIT_HF_CACHE=/home/comp/23481501/datasets/hf_cache
export SEEDMIT_OUTPUT_BASE=/home/comp/23481501/datasets/SeedMitigating_GPU/output_gpu/paper_reproduction
export SEEDMIT_CUDA_MODULE=cuda/11.4
export SEEDMIT_NUM_GPUS=2

mkdir -p /home/comp/23481501/datasets/SeedMitigating_GPU/output
cd /home/comp/23481501/datasets/SeedMitigating_GPU

bash gpu/run_full_training.sh

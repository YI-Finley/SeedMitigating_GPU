# GPU Environment Setup

This port is meant for a persistent user-owned environment on your SLURM cluster.

## Why A Separate Environment

Your current cluster setup is mixing:

- system/base Python
- user-installed `~/.local` packages

That pattern is fragile for PyTorch and often causes `import torch` to be killed.

This project assumes:

- a dedicated virtualenv or conda env
- `PYTHONNOUSERSITE=1`
- PyTorch installed inside that environment, not in `~/.local`

## Recommended Layout

- Project: `/home/comp/23481501/datasets/SeedMitigating_GPU`
- Virtualenv: `/home/comp/23481501/venvs/seedmitigating-gpu`
- Hugging Face cache: `/home/comp/23481501/datasets/hf_cache`
- Outputs: `/home/comp/23481501/datasets/SeedMitigating_GPU/output_gpu`

## Suggested Install Steps

```bash
module load cuda/11.4

python3.10 -m venv /home/comp/23481501/venvs/seedmitigating-gpu
source /home/comp/23481501/venvs/seedmitigating-gpu/bin/activate

export PYTHONNOUSERSITE=1

python -m pip install --upgrade pip setuptools wheel

# Install a CUDA-enabled PyTorch build that matches your cluster policy.
# Example only; adjust if your site provides a preferred wheel index.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

cd /home/comp/23481501/datasets/SeedMitigating_GPU
pip install -r requirements-gpu.txt
pip install -e ./verl
```

## Quick Verification

```bash
source /home/comp/23481501/venvs/seedmitigating-gpu/bin/activate
export PYTHONNOUSERSITE=1

python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.device_count())"
python -c "import transformers, ray; print(transformers.__version__); print(ray.__version__)"
python -c "import vllm; print(vllm.__version__)"
```

## If `import torch` Still Dies

- Make sure the environment is activated before running Python
- Make sure `PYTHONNOUSERSITE=1` is set
- Check that `pip show torch` points inside your virtualenv
- Check that `which python` also points inside your virtualenv
- Avoid using `base` while running this project

#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --job-name=kvcomm-smoke
#SBATCH --output=logs/kvcomm-%j.out
#SBATCH --error=logs/kvcomm-%j.err

set -euo pipefail

module load anaconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate kvcomm_qwen

# Show which Python the shell selected.
python -u -c 'import sys; print("Shell Python:", sys.executable)'

# Use the exact environment path from your conda list output.
KVCOMM_PYTHON="/tc1home/FYP/dtuan001/.conda/envs/kvcomm_qwen/bin/python"

"$KVCOMM_PYTHON" -u - <<'PY'
import sys
print("Explicit Python:", sys.executable)

import torch
print("PyTorch:", torch.__version__)
print("PyTorch location:", torch.__file__)
PY

cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

# These checks run on the allocated compute node.
hostname
nvidia-smi

export NLTK_DATA="$HOME/nltk_data_kvcomm"
"$KVCOMM_PYTHON" -u - <<'PY'
import torch
import transformers

print("PyTorch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("Transformers:", transformers.__version__)
assert torch.cuda.is_available(), "CUDA unavailable"
print("GPU:", torch.cuda.get_device_name(0))

x = torch.ones((32, 32), device="cuda", dtype=torch.float16)
print("FP16 GPU check:", (x @ x).mean().item())
PY


"$KVCOMM_PYTHON" -m pip freeze > "logs/environment-${SLURM_JOB_ID}.txt"

# Run all six methods sequentially using the same models and examples:
# skyline, baseline, KVComm, AC, NLD, then CIPHER.
"$KVCOMM_PYTHON" -u - \
  --model_A Qwen/Qwen3.5-9B \
  --model_B Qwen/Qwen3.5-9B \
  --device cuda:0 \
  --test_task tipsheets \
  --do_test_skyline \
  --do_test_baseline \
  --do_test \
  --do_test_ac \
  --do_test_nld \
  --do_test_cipher \
  --top_layers 0.3 \
  --calib_size 1 \
  --limit 2 \
  --seed 42 \
  --run_name "smoke-${SLURM_JOB_ID}" <<'PY'
import runpy

import torch

# Keep the experiment and memory counters in the same Python process.
try:
    runpy.run_path("com.py", run_name="__main__")
finally:
    # Also report peaks after a Python exception, including CUDA out-of-memory.
    if torch.cuda.is_initialized():
        gib = 1024 ** 3
        print("\nPeak GPU memory for the full smoke test (cuda:0):", flush=True)
        print(f"  PyTorch allocated: {torch.cuda.max_memory_allocated(0) / gib:.3f} GiB", flush=True)
        print(f"  PyTorch reserved:  {torch.cuda.max_memory_reserved(0) / gib:.3f} GiB", flush=True)
        print("  Reserved includes PyTorch's cache; CUDA context and non-PyTorch allocations are excluded.", flush=True)
    else:
        print("\nPeak GPU memory unavailable: CUDA was not initialized.", flush=True)
PY

#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --job-name=kvcomm-calibration-compare
#SBATCH --output=logs/kvcomm-calibration-compare-%j.out
#SBATCH --error=logs/kvcomm-calibration-compare-%j.err

# Submit from the repository root:
#   mkdir -p logs
#   sbatch run_calibration_comparison.sbatch

set -euo pipefail

module load anaconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate kvcomm_cf

# Show which Python the shell selected.
python -u -c 'import sys; print("Shell Python:", sys.executable)'

# Use the exact environment path from your conda list output.
KVCOMM_PYTHON="/tc1home/FYP/dtuan001/.conda/envs/kvcomm_cf/bin/python"

"$KVCOMM_PYTHON" -u - <<'PY'
import sys
print("Explicit Python:", sys.executable)

import torch
print("PyTorch:", torch.__version__)
print("PyTorch location:", torch.__file__)
PY

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
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

# Compare the two calibration methods using identical experiment settings.
# Both use one calibration example and 20 final-evaluation examples.
# The original method uses fixed alpha=1.0; the new method searches distinct sets.
# Layer 0 uses one selection slot in both runs, following com.py's default.
# Separate Python processes release the first model pair before the second run.

echo "[1/2] Original calibration"
"$KVCOMM_PYTHON" -u com.py \
  --model_A meta-llama/Llama-3.2-3B-Instruct \
  --model_B meta-llama/Llama-3.2-3B-Instruct \
  --device cuda:0 \
  --test_task tipsheets \
  --do_test \
  --top_layers 0.3 \
  --calib_size 1 \
  --limit 20 \
  --alpha 1.0 \
  --seed 42 \
  --run_name "calibration-original-${SLURM_JOB_ID}" \
  2>&1 | tee "logs/calibration-original-${SLURM_JOB_ID}.out"

echo "[2/2] Breakpoint calibration"
"$KVCOMM_PYTHON" -u com.py \
  --model_A meta-llama/Llama-3.2-3B-Instruct \
  --model_B meta-llama/Llama-3.2-3B-Instruct \
  --device cuda:0 \
  --test_task tipsheets \
  --do_test \
  --top_layers 0.3 \
  --calib_size 1 \
  --limit 20 \
  --breakpoint_calibration \
  --seed 42 \
  --run_name "calibration-breakpoint-${SLURM_JOB_ID}" \
  2>&1 | tee "logs/calibration-breakpoint-${SLURM_JOB_ID}.out"

echo "Comparison complete. Check the calibration-original and calibration-breakpoint logs for job ${SLURM_JOB_ID}."
echo "Snapshot logs and breakpoint candidate scores are saved under snapshots/."

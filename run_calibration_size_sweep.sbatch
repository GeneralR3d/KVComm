#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --job-name=kvcomm-calibration-size-sweep
#SBATCH --output=logs/kvcomm-calibration-size-sweep-%j.out
#SBATCH --error=logs/kvcomm-calibration-size-sweep-%j.err

# Submit from the repository root:
#   mkdir -p logs
#   sbatch run_calibration_size_sweep.sbatch
# All 20 runs share the time allocation above; increase it at submission if needed
# and permitted by the partition: sbatch --time=24:00:00 run_calibration_size_sweep.sbatch

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

# Compare both calibration methods for each calibration size from 1 through 10.
# Keep Qwen2.5-3B, Qasper, 50 final-evaluation examples, top_layers=0.3, and seed=42.
# The original method uses fixed alpha=0.8; breakpoint calibration searches distinct sets.
# Layer 0 uses one selection slot in both runs, following com.py's default.
# Separate Python processes release each model pair before the next run.
# Results from all 20 runs append to one CSV and JSON, including calib_size and method.
echo "Running 20 comparisons: 2 methods for each calibration size from 1 through 10."

for calib_size in {1..10}; do
  echo "[1/2] Original calibration (calib_size=$calib_size)"
  "$KVCOMM_PYTHON" -u com.py \
    --model_A Qwen/Qwen2.5-3B-Instruct\
    --model_B Qwen/Qwen2.5-3B-Instruct\
    --device cuda:0 \
    --test_task hotpotqa \
    --do_test \
    --top_layers 0.3 \
    --calib_size "$calib_size" \
    --limit 50 \
    --alpha 0.8 \
    --seed 42 \
    --results_csv "logs/calibration-size-sweep-${SLURM_JOB_ID}.csv" \
    --results_json "logs/calibration-size-sweep-${SLURM_JOB_ID}.json" \
    --run_name "calibration-original-size-${calib_size}-${SLURM_JOB_ID}" \
    2>&1 | tee "logs/calibration-original-size-${calib_size}-${SLURM_JOB_ID}.out"

  echo "[2/2] Breakpoint calibration (calib_size=$calib_size)"
  "$KVCOMM_PYTHON" -u com.py \
    --model_A Qwen/Qwen2.5-3B-Instruct\
    --model_B Qwen/Qwen2.5-3B-Instruct\
    --device cuda:0 \
    --test_task hotpotqa \
    --do_test \
    --top_layers 0.3 \
    --calib_size "$calib_size" \
    --limit 50 \
    --breakpoint_calibration \
    --seed 42 \
    --results_csv "logs/calibration-size-sweep-${SLURM_JOB_ID}.csv" \
    --results_json "logs/calibration-size-sweep-${SLURM_JOB_ID}.json" \
    --run_name "calibration-breakpoint-size-${calib_size}-${SLURM_JOB_ID}" \
    2>&1 | tee "logs/calibration-breakpoint-size-${calib_size}-${SLURM_JOB_ID}.out"

done

echo "Calibration-size sweep complete for job ${SLURM_JOB_ID}."
echo "Final comparison results: logs/calibration-size-sweep-${SLURM_JOB_ID}.csv and logs/calibration-size-sweep-${SLURM_JOB_ID}.json"
echo "Per-run logs include the calibration method and size in their filenames."
echo "Snapshot logs and breakpoint candidate scores are saved under snapshots/."

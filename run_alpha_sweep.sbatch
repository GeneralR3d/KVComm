#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --job-name=kvcomm-alpha-sweep
#SBATCH --output=logs/kvcomm-alpha-sweep-%j.out
#SBATCH --error=logs/kvcomm-alpha-sweep-%j.err

# Submit from the repository root:
#   mkdir -p logs
#   sbatch run_alpha_sweep.sbatch
# The 77 runs share the time allocation above. Increase it at submission if
# your partition permits, for example: sbatch --time=24:00:00 run_alpha_sweep.sbatch

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

# One CSV per job, appended immediately after each successful evaluation.
export KVCOMM_RESULTS_CSV="logs/alpha-sweep-${SLURM_JOB_ID}-results.csv"
echo "Communication results CSV: $KVCOMM_RESULTS_CSV"

# Seven datasets x eleven alphas = 77 evaluation cases.
# Smoke-test two examples per case; calibration uses one example per dataset.
TEST_TASKS=(countries tipsheets hotpotqa qasper musique multifieldqa_en twowikimqa)
ALPHAS=(1.0 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1 0.0)

# One process loads the model pair once and calibrates once per dataset.
# The runner saves successful scores, per-case memory peaks, and failure details.
total_runs=$(( ${#TEST_TASKS[@]} * ${#ALPHAS[@]} ))
echo "Running $total_runs cases with one model pair; evaluation limit=2 per case."

"$KVCOMM_PYTHON" -u com_sweep.py \
  --model_A Qwen/Qwen2.5-3B-Instruct \
  --model_B Qwen/Qwen2.5-3B-Instruct \
  --device cuda:0 \
  --test_tasks "${TEST_TASKS[@]}" \
  --alphas "${ALPHAS[@]}" \
  --top_layers 0.7 \
  --calib_size 1 \
  --max_input_length 8192 \
  --limit 2 \
  --seed 42 \
  --results_csv "$KVCOMM_RESULTS_CSV" \
  --run_name "alpha-sweep-${SLURM_JOB_ID}"

echo "Sweep finished. Check $KVCOMM_RESULTS_CSV and its JSONL companion for case results and failures."

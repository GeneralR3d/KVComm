#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --job-name=kvcomm-top-layers-sweep
#SBATCH --output=logs/kvcomm-top-layers-sweep-%j.out
#SBATCH --error=logs/kvcomm-top-layers-sweep-%j.err

# Submit from the repository root:
#   mkdir -p logs
#   sbatch run_top_layers_sweep.sbatch
# The 21 evaluation cases share the time allocation above. Increase it at submission if
# your partition permits, for example: sbatch --time=24:00:00 run_top_layers_sweep.sbatch

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

# Seven datasets x three top-layer fractions = 21 evaluation cases.
# Evaluate full datasets (--limit 0); calibration uses one example per dataset.
TEST_TASKS=(countries tipsheets hotpotqa qasper musique multifieldqa_en twowikimqa)
# Edit this array to choose the fractions of layers to share.
TOP_LAYERS=(0.3 0.5 0.7)

# Each fraction starts a fresh process that loads one model pair and calibrates
# once per dataset. Alpha is fixed at one, so selection uses calibrated importance.
total_runs=$(( ${#TEST_TASKS[@]} * ${#TOP_LAYERS[@]} ))
echo "Running $total_runs cases across ${#TOP_LAYERS[@]} top-layer fractions; full-dataset evaluation."

for top_layers in "${TOP_LAYERS[@]}"; do
  # One CSV and JSONL companion per fraction, isolated by Slurm job ID.
  export KVCOMM_RESULTS_CSV="logs/top-layers-sweep-${SLURM_JOB_ID}-top-layers-${top_layers}-results.csv"
  echo "Starting top_layers=$top_layers; communication results CSV: $KVCOMM_RESULTS_CSV"

  "$KVCOMM_PYTHON" -u com_sweep.py \
    --model_A meta-llama/Llama-3.2-3B-Instruct \
    --model_B meta-llama/Llama-3.2-3B-Instruct \
    --device cuda:0 \
    --test_tasks "${TEST_TASKS[@]}" \
    --alphas 1.0 \
    --top_layers "$top_layers" \
    --calib_size 1 \
    --limit 0 \
    --seed 42 \
    --results_csv "$KVCOMM_RESULTS_CSV" \
    --run_name "top-layers-sweep-${SLURM_JOB_ID}-top-layers-${top_layers}"

  echo "Finished top_layers=$top_layers. Check $KVCOMM_RESULTS_CSV and its JSONL companion for case results and failures."
done

echo "Top-layer sweep finished: $total_runs evaluation cases attempted. Check each fraction's CSV and JSONL diagnostics for outcomes."

#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --job-name=kvcomm-baseline-skyline
#SBATCH --output=logs/kvcomm-baseline-skyline-%j.out
#SBATCH --error=logs/kvcomm-baseline-skyline-%j.err

# Submit from the repository root:
#   mkdir -p logs
#   sbatch run_baseline_skyline.sbatch
# The 14 evaluation cases share the time allocation above. Increase it at submission if
# your partition permits, for example: sbatch --time=24:00:00 run_baseline_skyline.sbatch

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

# Seven datasets x two methods = 14 evaluation cases.
# Baseline uses receiver input only; skyline includes sender context as text.
# Both methods evaluate models A and B, as implemented by com.py.
# Evaluate full datasets (--limit 0) using com.py's default input cap.
TEST_TASKS=(countries tipsheets hotpotqa qasper musique multifieldqa_en twowikimqa)

total_runs=$(( ${#TEST_TASKS[@]} * 2 ))
echo "Running baseline and skyline: $total_runs cases; full-dataset evaluation."

for task in "${TEST_TASKS[@]}"; do
  # Each dataset starts a fresh process and loads the model pair once.
  # com.py runs skyline first, then baseline, and logs both scores separately.
  case_log="logs/baseline-skyline-${SLURM_JOB_ID}-${task}.out"
  echo "Starting task=$task; output log: $case_log"

  "$KVCOMM_PYTHON" -u com.py \
    --model_A google/gemma-3-1b-it \
    --model_B google/gemma-3-1b-it \
    --device cuda:0 \
    --test_task "$task" \
    --do_test_baseline \
    --do_test_skyline \
    --limit 0 \
    --seed 42 \
    --run_name "baseline-skyline-${SLURM_JOB_ID}-${task}" \
    2>&1 | tee "$case_log"

  echo "Finished task=$task. Baseline and skyline scores are in $case_log and the snapshot log."
done

echo "Baseline and skyline finished: $total_runs evaluation cases completed."

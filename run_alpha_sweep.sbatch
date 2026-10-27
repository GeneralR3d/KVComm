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

# Exact --test_task names from dataloader.get_evaluator (seven datasets).
# The standard hotpotqa, qasper, and musique evaluators sample 500 examples;
# --limit below controls evaluation size; 0 uses every supplied example.
TEST_TASKS=(countries tipsheets hotpotqa qasper musique multifieldqa_en twowikimqa)
ALPHAS=(1.0 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1 0.0)

# Run KVComm only, sequentially in one allocation. A fresh Python process for
# each pair releases model/GPU memory and keeps per-run peak counters separate.
# Keep calibration size, seed, and all other settings fixed across alpha values.
# Each pair gets its own snapshots/alpha-sweep-<job>-<task>-alpha-<value>_* folder.
run_index=0
total_runs=$(( ${#TEST_TASKS[@]} * ${#ALPHAS[@]} ))
for test_task in "${TEST_TASKS[@]}"; do
  for alpha in "${ALPHAS[@]}"; do
    run_index=$((run_index + 1))
    echo "[$run_index/$total_runs] task=$test_task alpha=$alpha KVComm=0.7"
    "$KVCOMM_PYTHON" -u - \
      --model_A Qwen/Qwen3-4B \
      --model_B Qwen/Qwen3-4B \
      --device cuda:0 \
      --test_task "$test_task" \
      --do_test \
      --top_layers 0.7 \
      --alpha "$alpha" \
      --calib_size 1 \
      --limit 0 \
      --seed 42 \
      --run_name "alpha-sweep-${SLURM_JOB_ID}-${test_task}-alpha-${alpha}" <<'PY'
import csv
import os
from pathlib import Path
import runpy

import torch

# Keep the experiment and memory counters in the same Python process.
try:
    # main() returns the final evaluation score after layer calibration.
    # Loading without __main__ lets us capture that value directly.
    experiment = runpy.run_path("com.py")
    config = experiment["parse_args"]()
    communication_result = experiment["main"](config)

    # Close the CSV after each row so completed runs survive a later failure.
    results_path = Path(os.environ["KVCOMM_RESULTS_CSV"])
    with results_path.open("a", newline="") as output:
        writer = csv.writer(output)
        if output.tell() == 0:
            writer.writerow(["test_task", "alpha", "communication_result"])
        writer.writerow([config.test_task, config.alpha, float(communication_result)])
    print(f"Saved communication result to {results_path}: {communication_result}", flush=True)
finally:
    # Also report peaks after a Python exception, including CUDA out-of-memory.
    if torch.cuda.is_initialized():
        gib = 1024 ** 3
        print("\nPeak GPU memory for the current dataset/alpha run (cuda:0):", flush=True)
        print(f"  PyTorch allocated: {torch.cuda.max_memory_allocated(0) / gib:.3f} GiB", flush=True)
        print(f"  PyTorch reserved:  {torch.cuda.max_memory_reserved(0) / gib:.3f} GiB", flush=True)
        print("  Reserved includes PyTorch's cache; CUDA context and non-PyTorch allocations are excluded.", flush=True)
    else:
        print("\nPeak GPU memory unavailable: CUDA was not initialized.", flush=True)
PY
  done
done

echo "Completed all $total_runs dataset/alpha runs."

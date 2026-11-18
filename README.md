# KVComm

Official implementation of the paper [KVComm: Enabling Efficient LLM Communication through Selective KV Sharing](https://openreview.net/forum?id=F7rUng23nw) (ICLR 2026).

A framework for communicating between Large Language Models (LLMs), focusing on how models can effectively share information to improve collaborative reasoning and question-answering performance.

## Installation

```bash
pip install -r requirements.txt
```

Note: Requires `transformers==4.53.3` specifically.

## Datasets

| Dataset           | Task Type             | Description               | Data Path                         |
|-------------------|-----------------------|---------------------------|-----------------------------------|
| `hotpotqa`        | Multi-hop QA          | Wikipedia-based reasoning | HuggingFace                       |
| `qasper`          | Scientific QA         | Paper-based questions     | HuggingFace                       |
| `musique`         | Multi-hop QA          | Compositional reasoning   | HuggingFace                       |
| `multifieldqa_en` | Multi-domain QA       | Cross-field knowledge     | HuggingFace                       |
| `twowikimqa`      | Multi-hop QA          | Wikipedia bridge entities | HuggingFace                       |
| `tipsheets`       | Custom QA             | Synthetic reasoning tasks | `dataloader/data/tipsheets.jsonl` |
| `countries`       | Geographic QA         | Country-based questions   | `dataloader/data/countries.jsonl` |
| `tmath`           | Mathematical          | Math problem solving      | `dataloader/data/TMATH`           |

## Quick Start

### Baseline Test
```bash
python com.py \
    --test_task hotpotqa \
    --do_test_baseline \
    --model_A meta-llama/Llama-3.1-8B-Instruct \
    --model_B meta-llama/Llama-3.1-8B-Instruct
```

### Skyline Test
```bash
python com.py \
    --test_task hotpotqa \
    --do_test_skyline \
    --model_A meta-llama/Llama-3.1-8B-Instruct \
    --model_B meta-llama/Llama-3.1-8B-Instruct
```

### KVComm Communication
```bash
python com.py \
    --test_task hotpotqa \
    --do_test \
    --model_A meta-llama/Llama-3.1-8B-Instruct \
    --model_B meta-llama/Llama-3.1-8B-Instruct \
    --top_layers 0.3
```

### Activation Communication
```bash
python com.py \
    --test_task tipsheets \
    --do_test_ac \
    --model_A meta-llama/Llama-3.1-8B-Instruct \
    --model_B meta-llama/Llama-3.1-8B-Instruct \
    --layer_k 26 \
    --layer_j 26 \
    --f replace
```

### Natural Language Debate
```bash
python com.py \
    --test_task hotpotqa \
    --do_test_nld \
    --model_A meta-llama/Llama-3.1-8B-Instruct \
    --model_B meta-llama/Llama-3.1-8B-Instruct \
    --nld_max_tokens_model_A_and_B_phase1 256 \
    --sender_aware
```

### CIPHER Communication
```bash
python com.py \
    --test_task hotpotqa \
    --do_test_cipher \
    --model_A meta-llama/Llama-3.1-8B-Instruct \
    --model_B meta-llama/Llama-3.1-8B-Instruct \
    --nld_max_tokens_model_A_and_B_phase1 256 \
    --sender_aware
```

## Communication Methods

### 1. KVComm (Cross-View Communication)
- **Mechanism**: Shares key-value cache from model A's specified layers to model B
- **Parameters**: `--layers_list`, `--layer_from`, `--layer_to`, `--top_layers`
- **Use Case**: Efficient information transfer with minimal computational overhead

### 2. Activation Communication (AC)
- **Mechanism**: Injects hidden activations from model A into model B at specific layers
- **Parameters**: `--layer_k` (source), `--layer_j` (target), `--f` (fusion method)
- **Fusion Methods**: `replace`, `sum`, `mean`

### 3. Natural Language Debate (NLD)
- **Mechanism**: Models exchange natural language responses and refine answers
- **Parameters**: `--nld_max_tokens_model_A_and_B_phase1`, `--sender_aware`
- **Process**: Initial responses → Exchange → Refinement

### 4. CIPHER Communication
- **Mechanism**: Models communicate through learned embedding representations
- **Features**: Temperature-controlled generation, nearest neighbor decoding

## Configuration Options

### Model Configuration
- `--model_A`, `--model_B`: Hugging Face model identifiers
- `--device`: CUDA device (default: `cuda:0`)
- `--max_input_length`: Maximum input token length (default: 64000)

### Communication Parameters
- `--layers_list`: Specific layers for KVComm communication
- `--top_layers`: Percentage of top-importance layers to use
- `--layer_k`, `--layer_j`: Source and target layers for AC
- `--f`: Fusion function for AC (`replace`, `sum`, `mean`)

### Evaluation Settings
- `--test_task`: Dataset to evaluate on
- `--limit`: Limit number of evaluation examples
- `--calib_size`: Calibration set size for layer importance

### Experiment Tracking
- `--use_wandb`: Enable Weights & Biases logging
- `--wandb_project`: W&B project name
- `--wandb_entity`: W&B entity
- `--run_name`: Custom experiment name

## Layer Importance Analysis

The framework includes automatic layer importance detection:

```bash
python com.py \
    --test_task hotpotqa \
    --do_test \
    --top_layers 0.3
```

This automatically identifies which layers are most important for communication and selects them for the main evaluation.

### Breakpoint calibration

```bash
python -u com.py --do_test --test_task qasper --breakpoint_calibration --top_layers 0.3 --calib_size 1 --limit 0
```

`--breakpoint_calibration` is off by default; omit it to use the existing
fixed-`--alpha` selection and task metric. In `com.py`, enabling the flag uses
`--calib_size` examples for attention tracing and the same `--calib_size`
examples for each candidate set's mean answer F1. Both calibration stages use
at most the available dataset size. F1 ties keep the first alpha interval.
The winning layers are retained in the configuration, then evaluated once in
a separate final run using the task's normal metric. Only this final run uses
`--limit`: `0` uses all examples loaded by the task adapter, while a positive
value caps the final run. For example, `--calib_size 5 --limit 100` traces 5
examples, scores each candidate on those 5 examples, then evaluates the winner
on up to 100 examples.

In `com.py`, layer 0 reserves one slot by default for breakpoint, fixed-alpha,
random, and layer-curve selection. Use `--no_always_include_layer_0` to remove
that selection constraint. The communicator itself continues to share layer 0
as before. Explicit `--layers_list` selections are preserved.

Each run directory contains `calibration_candidates.csv` (midpoint alpha,
exact alpha fraction, selected layers, answer F1) and
`calibration_selection.json` (winner and candidate count). Candidate errors
stop the run, preserve completed CSV rows, and do not produce a winner.
Breakpoint calibration requires `--do_test` and a positive `--top_layers`
budget of at least one layer; it cannot be combined with `--random_selection`
or `--do_layer_curve`.

Offline CPU regression tests (including tiny local Llama attention tracing):
`python -m pytest -q tests/test_com.py`.

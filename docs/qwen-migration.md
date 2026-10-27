# Qwen on Transformers 5.21.0

This runtime supports dense **Qwen2 / Qwen2.5**, dense **Qwen3**, and
**Qwen3 MoE**, using full-attention KV caches. The standard, multiple-sender,
and online entry points select it from the model configuration, including
local checkpoint directories. Llama and Gemma retain their original code
and `transformers==4.53.3` environment.

## Install separately

Use Python 3.11 or newer and a separate environment. Install a PyTorch build
appropriate for your CUDA driver before installing the remaining requirements.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-qwen.txt
```

Do not install `requirements.txt` into this environment afterward: it pins
the legacy version. Keep the legacy environment separately if you use it.

If cloning an existing environment, upgrade both pinned packages:

```bash
python -m pip install --upgrade "transformers==5.21.0" "peft==0.21.2"
python -m pip check
python -c 'from transformers.trainer_utils import set_seed; import transformers, peft; print(transformers.__version__, peft.__version__)'
```

An older PEFT can fail on import with `cannot import name 'HybridCache'`.
Transformers imports installed PEFT from `trainer_utils`, which the entry
points use for `set_seed`, even though these experiments do not train adapters.
The original CPU test environment had no PEFT installed, so the initial
verification did not expose this cloned-environment incompatibility.

## Run a small comparison

From the repository root, run the same checkpoint, dataset, and seed for the
baseline and communication paths. This example uses two copies of Qwen3-8B;
choose a smaller supported checkpoint if GPU memory is limited.

```bash
python com.py \
  --model_A Qwen/Qwen3-8B --model_B Qwen/Qwen3-8B \
  --test_task tipsheets --do_test_baseline --do_test \
  --top_layers 0.3 --calib_size 1 --limit 2 --seed 42
```

For MoE, replace both model identifiers with a Qwen3 MoE checkpoint such as
`Qwen/Qwen3-30B-A3B`. Budget for the complete weights of each model: the
active parameter count is not the total weight memory requirement.

For manually selected layers, replace `--top_layers 0.3 --calib_size 1` with
`--layers_list 0 8 16`. Layer indices are zero-based and must exist in the
selected model. Layer 0 always transfers its full cache. Unselected layers
retain only the first sender token, plus subsequently processed receiver tokens.

`com_ms.py` uses the same runtime with `--model_A1`, `--model_A2`, and
`--model_B`. `com_online.py` uses it for periodic calibration. Their task
options remain unchanged. Use identical checkpoints initially; the runtime
requires matching architecture, layer count, and KV-head geometry, and does
not learn alignment between different checkpoints. The existing evaluator
continues to use the receiver tokenizer for all models.

For activation communication, use `--do_test_ac --layer_k 16 --layer_j 16`
with valid indices for your model. Negative indices count from the end.
`layer_k` indexes the sender's hidden-state tuple; `layer_j` identifies the
receiver decoder layer whose input is modified, as in the original method.

Qwen3 thinking is disabled by default to suit the existing short-answer
evaluation. `--qwen_enable_thinking` enables the model's chat-template
thinking mode. The evaluators retain their existing generation budgets,
greedy decoding, and answer extraction; this flag alone does not establish
a calibrated reasoning-model benchmark.

## What changed and why

- `model_runtime.py` routes Qwen lazily. Importing the entry points no longer
  imports legacy Llama/Gemma attention implementations into the Qwen environment.
  Qwen loads with the v5 `dtype` argument; legacy loading keeps `torch_dtype`.
- `models_qwen.py` reads `cache.layers[i].keys` and `.values`, creates a new
  receiver cache, and leaves the donor cache unchanged. Single-token receiver
  prompts also initialize transfer correctly.
- Qwen attention uses a registered SDPA backend. Native Q/K projections,
  Q/K normalization, RoPE, decoder layers, and MoE expert routing are preserved.
  Calibration captures the actual Q and K presented to that backend, rather
  than maintaining copies of each architecture's attention forward method.
- Full-cache and sink-cache layers have different sequence lengths. Their
  masks retain the corresponding donor columns and receiver columns, so
  receiver tokens remain causal. RoPE positions follow the full layer-0 cache;
  no `shift_back` behavior is implemented.
- Activation communication uses a temporary decoder input hook, removed even
  on failure, rather than copying Transformers decoder loops.
- A tokenizer adapter preserves the tensor output expected by the existing
  evaluators despite v5 tokenizer return defaults, and sets thinking mode.

## Limits

`shift_back` is explicitly rejected on this runtime. Sliding-window,
hybrid/recurrent, quantized/static caches, and vision architectures are not
integrated. **Qwen3.5 is not supported by this migration**: its hybrid state
needs a separate transfer design. The existing `run_smoke.sbatch` currently
names Qwen3.5-9B and must be adapted to a supported checkpoint and the Qwen
environment before use; this migration does not edit that job script.

Tests cover the repository's single-beam cached generation flows. GPU kernels,
full pretrained checkpoints, dataset accuracy, and throughput still require
validation on the target hardware. Attention calibration materializes attention
matrices and can consume substantially more memory than ordinary SDPA inference.

## Verification

Install pytest and run the offline CPU suite. Models are tiny, randomly
initialized real Qwen implementations; no checkpoint or model tokenizer is
downloaded. Entry-point import tests suppress the pre-existing NLTK corpus
downloads, which are unrelated external I/O.

```bash
python -m pip install pytest
HF_HUB_OFFLINE=1 python -m pytest tests -q
```

The suite checks native forward/generation parity, dense and MoE checkpoint
loading, cache contents and donor immutability, causal masks on sink layers,
decode cache growth, tracing, repeated calibration, two senders, activation
injection, tokenizer behavior, CLI imports, and invalid configurations.

During development: macOS ARM64 CPU, Python 3.13.5, PyTorch 2.14.1,
Transformers 5.21.0. All 67 tests also passed with PEFT 0.21.2 installed,
after reproducing the `HybridCache` import failure with PEFT 0.17.1.
No CUDA GPU was available. These tests establish software compatibility,
not pretrained-model communication accuracy.

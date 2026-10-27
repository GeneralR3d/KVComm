# Qwen-only Transformers migration

User scope: migrate Qwen on a new branch; include dense Qwen2/Qwen2.5,
dense Qwen3, and Qwen3 MoE. Do not modify Llama or Gemma implementations.
Do not implement `shift_back`.

## Approach

Use an isolated Qwen runtime on Transformers 5.21.0, the published stable
release checked on 2026-10-27. Keep `requirements.txt` and all existing
`models*.py` / `model_attn.py` implementations unchanged. Add a separate
Qwen requirements file and select the new runtime lazily from the three
entry points. Legacy models continue to use their existing 4.53.3 environment.

A global upgrade would break legacy internals and is outside scope.
Maintaining parallel copied forward implementations would duplicate upstream
logic; instead capture attention inputs at the attention backend boundary
and use decoder pre-hooks for activation communication.

## Behavior

- Preserve selected-layer transfer, full layer-0 transfer, and the first-token
  attention sink at other layers, using modern per-layer cache storage.
- Support single sender, two senders, online calibration, and Qwen activation
  communication without replacing upstream decoder implementations.
- Validate model family, KV dimensions, layer alignment, selected indices,
  and supported cache/attention types before transfer. Initially require
  equal layer counts; heterogeneous representation alignment is not added.
- Preserve native Q/K normalization, rotary embeddings, and MoE routing.
- Make Qwen3 thinking mode an explicit CLI option, defaulting off for the
  existing short-answer evaluation. Preserve legacy prompting.
- Reject hybrid/recurrent and vision architectures rather than treating
  their state as a conventional KV cache. Qwen3.5 is outside this scope.

## Verification

Use tiny randomly initialized real Qwen2, Qwen3, and Qwen3 MoE models on CPU;
no checkpoint download is needed. Test numerical equivalence with native
forward/generation when all layers transfer, exact selected-cache contents,
prefill and decoding, tracing parity and attention capture, multi-sender
concatenation, activation injection, CLI routing, and invalid inputs.
Run the full new test suite and import/CLI checks. GPU dataset accuracy,
latency, and memory measurements require a later CUDA run.

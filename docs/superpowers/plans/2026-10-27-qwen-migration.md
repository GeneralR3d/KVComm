# Qwen Migration Implementation Plan

> Execution: implement inline in the requested branch using test-first changes.

**Goal:** Run Qwen2/2.5, Qwen3, and Qwen3 MoE with Transformers 5.21.0.

**Architecture:** A separate Qwen communicator and attention capture layer;
lazy routing in existing entry points preserves the legacy implementations.

**Tech stack:** Python, PyTorch, Transformers 5.21.0, pytest.

**Spec:** `docs/superpowers/specs/2026-10-27-qwen-migration-design.md`.

## Constraints

- Leave `models.py`, `models_ms.py`, `models_ac.py`, `model_attn.py`,
  and the original requirements pin unchanged.
- No `shift_back`; no Qwen3.5/hybrid/vision state transfer.
- No remote checkpoint downloads for tests; no claims of GPU evaluation.

## Review focus

- Native attention/generation output parity, including Qwen3 MoE routing.
- Different per-layer cache lengths and causal masking during decode.
- Repeated calibration must not stack tracers or retain stale inputs.
- Unsupported pairs and invalid layer indices fail clearly.
- Legacy entry points must not import migrated internals unnecessarily.

## Tasks

1. Write tests using tiny real Qwen models for cache transfer and full-cache
   native output parity. Run red, implement `models_qwen.py`, run green.
2. Add tests for attention capture, generation, multiple senders, and
   activation injection. Run red, implement the integrations, run green.
3. Test runtime selection and Qwen prompt options; route `com.py`,
   `com_ms.py`, and `com_online.py` lazily. Add separate requirements and
   usage documentation. Run the suite and CLI checks.
4. Review the complete diff, verify excluded implementation files are
   unchanged, and record dependency versions, results, and GPU limitations.

## Progress

- Branch created: `codex/qwen-transformers-migration`.
- Temporary CPU test environment installation started.
- Scope clarified: include Qwen3 MoE.
- Ruling: user explicitly requested beginning implementation; proceed within
  that scope without another design approval round.
- Task 1 complete: 39 failing tests before implementation; 39 passed after
  implementing selective cache transfer, generation, tracing, and multiple senders.
- Task 2 complete: 16 failures for missing activation/runtime functionality;
  all 58 tests passed after implementation. A wrapper-only expert configuration
  prevents dispatch checks on the wrapper; receiver expert dispatch is preserved.
- Task 3: CLI import checks reproduced legacy import failures under v5;
  lazy routing fixed all three entry points. Full suite at that point: 61 passed.
- Checkpoint loading and sliding-attention validation checks: 6 passed.
- Task 3 complete: full offline CPU suite: 67 passed in 10.98 seconds;
  syntax checks and `pip check` passed. Excluded legacy implementation files
  and `requirements.txt` match HEAD byte-for-byte.
- Final read-only review dispatched against the complete working changes.
- Final review complete: no actionable correctness findings. Reviewer reran
  all 67 tests and checked selective-cache prefill/decode parity with receiver
  padding for all three architectures.
- Verification limits: CUDA kernels, full pretrained checkpoints, benchmark
  accuracy, throughput and memory capacity remain unverified. Beam search
  and uncached generation are outside the documented evaluator flows.

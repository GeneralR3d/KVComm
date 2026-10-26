# Repository Guidelines

## Project Structure & Module Organization

KVComm studies LLM communication through selective key-value cache sharing.

- `com.py` runs standard experiments; `com_ms.py` handles multiple senders; `com_online.py` handles mixed-task online calibration.
- `models*.py` implement communication methods, `model_attn.py` traces attention, and `layer_importance.py` selects layers.
- `eval.py`, `eval_ms.py`, and `eval_online.py` implement evaluation flows.
- `dataloader/` contains task evaluators and factories in `__init__.py`; `dataloader/data/` holds bundled datasets, including TMATH.
- `utils/` provides scoring, logging, and run naming. Generated logs go under `snapshots/` by default.

## Build, Test, and Development Commands

Run commands from the repository root, preferably inside a virtual environment:

```bash
python -m pip install -r requirements.txt
python com.py --help
python com.py --test_task tipsheets --do_test_baseline --limit 2
python com.py --test_task tipsheets --do_test --top_layers 0.3 --limit 2
```

These install dependencies, show experiment options, run a small baseline, and exercise KV sharing with layer selection. There is no separate build step. Keep `transformers==4.53.3`: the implementation depends on its model internals. Experiments default to two Llama-3.1-8B-Instruct models on `cuda:0`; provision compatible PyTorch/CUDA, sufficient GPU memory, and Hugging Face model access first. Override models with `--model_A` and `--model_B`.

## Coding Style & Naming Conventions

Use four-space indentation, `snake_case` for modules and functions, and `PascalCase` for classes. Follow surrounding Python style; no formatter or linter is configured. Add experiment options to the entry point's `AlignConfig` dataclass, which supplies CLI arguments. Implement dataset adapters as `BaseEvaluator` subclasses and register them in `dataloader/__init__.py`.

## Testing Guidelines

No automated test framework or coverage threshold is configured. Validate changes with small `--limit` runs of the affected evaluation path and compare baseline and communication metrics using identical models, datasets, and `--seed`. Record commands, dependency versions, GPU details, and results. If GPU evaluation is unavailable, state that limitation explicitly. Name new automated tests `test_*.py` under `tests/` and document their runner.

## Commit & Pull Request Guidelines

History is limited to initial commits and a short descriptive update; no formal commit convention is established. Use concise, imperative subjects such as `Add Qasper evaluation support`. Keep changes focused. PRs should explain the behavior changed, link relevant issues, and include reproducible validation commands and metric comparisons where applicable. Keep credentials, model weights, and generated `snapshots/` or `wandb/` artifacts out of commits.

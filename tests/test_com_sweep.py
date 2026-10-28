"""Offline CPU checks: python -m pytest -q tests/test_com_sweep.py."""

import copy
import csv
import importlib
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import weakref

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM


@pytest.fixture
def sweep(monkeypatch):
    # Existing dataset imports download NLTK resources at module import time.
    import nltk
    monkeypatch.setattr(nltk, "download", lambda *args, **kwargs: True)
    return importlib.import_module("com_sweep")


@pytest.fixture
def model():
    torch.set_num_threads(1)
    torch.manual_seed(7)
    cfg = Qwen3Config(
        vocab_size=64, hidden_size=32, intermediate_size=48,
        num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, max_position_embeddings=128, attention_dropout=0.0,
        pad_token_id=0, bos_token_id=1, eos_token_id=63,
    )
    cfg._attn_implementation = "sdpa"
    return Qwen3ForCausalLM(cfg).eval()


class Tokenizer:
    eos_token_id = 63
    pad_token_id = 0

    def apply_chat_template(self, messages, **kwargs):
        words = messages[0]["content"].split()
        return torch.tensor([[1] + [2 + sum(map(ord, word)) % 50 for word in words]])

    def decode(self, ids, **kwargs):
        return " ".join(map(str, ids.tolist()))


class Dataset:
    name = "tipsheets"
    max_tokens = 2
    truncate_input = True

    def __init__(self):
        self.data = [
            {"id": "first", "prompt_A": "a b c", "prompt_B": "q"},
            {"id": "second", "prompt_A": "d e f", "prompt_B": "r"},
        ]
        self.f1_count = 0
        self.f1_total = 0.0

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        self.f1_count = 0
        self.f1_total = 0.0
        return iter(self.data)

    def evaluate_item(self, item, response):
        self.f1_count += 1
        self.f1_total += sum(map(int, response.split())) / 128

    def get_result(self):
        return self.f1_total / self.f1_count if self.f1_count else 0.0


@pytest.fixture
def offline(sweep, model, monkeypatch):
    loaded, datasets, tokenizers = [], [], []

    def load(*args, **kwargs):
        value = copy.deepcopy(model)
        loaded.append(value)
        return value

    def dataset(task):
        value = Dataset()
        value.name = task
        datasets.append(weakref.ref(value))
        return value

    def tokenizer(*args, **kwargs):
        tokenizers.append(Tokenizer())
        return tokenizers[-1]

    monkeypatch.setattr(sweep, "load_experiment_model", load)
    monkeypatch.setattr(sweep.AutoConfig, "from_pretrained", lambda *a, **k: copy.deepcopy(model.config))
    monkeypatch.setattr(sweep.AutoTokenizer, "from_pretrained", tokenizer)
    monkeypatch.setattr(sweep, "get_evaluator", dataset)
    return SimpleNamespace(loaded=loaded, datasets=datasets, tokenizers=tokenizers)


def config(sweep, tmp_path, **kwargs):
    defaults = dict(device="cpu", snapshot_path=str(tmp_path),
                    test_tasks=["tipsheets"], alphas=[1.0, 0.0],
                    max_input_length=96, limit=2, calib_size=1)
    defaults.update(kwargs)
    return sweep.AlignConfig(**defaults)


def records(tmp_path):
    return [json.loads(line) for path in tmp_path.rglob("*.jsonl")
            for line in path.read_text().splitlines()]


def csv_rows(tmp_path):
    paths = list(tmp_path.rglob("*.csv"))
    assert len(paths) == 1
    with paths[0].open() as file:
        reader = csv.DictReader(file)
        assert reader.fieldnames == ["test_task", "alpha", "communication_result"]
        return list(reader)


def test_default_sweep_and_cli_lists(sweep):
    cfg = sweep.parse_args([])
    assert len(cfg.test_tasks) == 7
    assert cfg.alphas == [i / 10 for i in range(10, -1, -1)]
    assert cfg.model_A == cfg.model_B == "Qwen/Qwen3-4B"
    assert cfg.max_input_length == 8192
    assert cfg.top_layers == 0.7
    cfg = sweep.parse_args(["--test_tasks", "qasper", "tipsheets", "--alphas", "0.7", "0.0"])
    assert cfg.test_tasks == ["qasper", "tipsheets"]
    assert cfg.alphas == [0.7, 0.0]


def test_help_does_not_load_datasets_or_models():
    result = subprocess.run([sys.executable, "-c", '''
import runpy, sys
sys.argv = ["com_sweep.py", "--help"]
try:
    runpy.run_path("com_sweep.py", run_name="__main__")
except SystemExit as error:
    assert error.code == 0
assert "dataloader" not in sys.modules
'''], cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True,
        env=dict(os.environ, HF_HUB_OFFLINE="1"))
    assert result.returncode == 0, result.stderr
    assert "--max_input_length" in result.stdout


@pytest.mark.parametrize("override", [
    {"alphas": [float("nan")]}, {"alphas": [1.1]}, {"sigma": 0},
    {"calib_size": 0}, {"limit": -1}, {"max_input_length": 0},
    {"test_tasks": ["unknown"]}, {"top_layers": 0},
])
def test_invalid_configuration_fails_before_loading(sweep, offline, tmp_path, override):
    with pytest.raises(ValueError):
        sweep.main(config(sweep, tmp_path, **override))
    assert not offline.loaded


@pytest.mark.parametrize("budget,expected", [(5, [0, 1, 2, 8, 9]), (4, [0, 1, 8, 9]), (1, [0])])
def test_middle_truncation_keeps_exact_budget(sweep, budget, expected):
    dataset = Dataset()
    ev = sweep.SweepCommunicationEvaluator(dataset, Tokenizer(), False, budget + 2)
    a, b = ev.truncate_input(torch.arange(10)[None], torch.tensor([[20, 21]]))
    assert a.tolist() == [expected]
    assert b.tolist() == [[20, 21]]
    assert ev.truncation_count == 1


@pytest.mark.parametrize("receiver_length", [5, 6])
def test_zero_or_negative_sender_budget_is_a_failure(sweep, receiver_length):
    ev = sweep.SweepCommunicationEvaluator(Dataset(), Tokenizer(), False, 5)
    with pytest.raises(sweep.InputLimitError):
        ev.truncate_input(torch.tensor([[1]]), torch.ones((1, receiver_length), dtype=torch.long))


def test_exact_limit_and_prohibited_truncation(sweep):
    dataset = Dataset()
    dataset.truncate_input = False
    ev = sweep.SweepCommunicationEvaluator(dataset, Tokenizer(), False, 5)
    a, b = torch.tensor([[1, 2, 3]]), torch.tensor([[4, 5]])
    assert ev.truncate_input(a, b)[0] is a
    with pytest.raises(sweep.InputLimitError):
        ev.truncate_input(torch.tensor([[1, 2, 3, 4]]), b)


def test_pair_loaded_once_and_calibration_once_per_dataset(sweep, offline, tmp_path):
    original_handlers = list(logging.getLogger().handlers)
    result = sweep.main(config(sweep, tmp_path, test_tasks=["tipsheets", "countries"]))
    assert len(offline.loaded) == 2
    assert offline.loaded[0] is not offline.loaded[1]
    assert len(offline.tokenizers) == 1
    assert len(result) == 4
    assert all(row["status"] == "success" and row["items_evaluated"] == 2 for row in result)
    assert len([row for row in records(tmp_path) if row["stage"] == "calibration"]) == 2
    assert len(csv_rows(tmp_path)) == 4
    assert logging.getLogger().handlers == original_handlers
    assert all(ref() is None for ref in offline.datasets)
    receiver = offline.loaded[1]
    assert receiver.config._attn_implementation == "sdpa"
    assert all(not getattr(block.self_attn, "_kvcomm_trace", False) for block in receiver.model.layers)
    assert all(getattr(block.self_attn, "attn_inputs", None) is None for block in receiver.model.layers)


def test_case_order_does_not_change_scores_or_selected_layers(sweep, offline, tmp_path):
    first = sweep.main(config(sweep, tmp_path / "first"))
    second = sweep.main(config(sweep, tmp_path / "second", alphas=[0.0, 1.0]))
    def values(rows):
        return {row["alpha"]: (row["communication_result"], row["selected_layers"]) for row in rows}
    assert values(first) == values(second)


def test_reused_calibration_matches_separate_single_alpha_runs(sweep, offline, tmp_path):
    combined = sweep.main(config(sweep, tmp_path / "combined"))
    for row in combined:
        single = sweep.main(config(sweep, tmp_path / str(row["alpha"]), alphas=[row["alpha"]]))[0]
        assert single["selected_layers"] == row["selected_layers"]
        assert single["communication_result"] == row["communication_result"]


def test_sender_limits_logits_without_changing_cache(sweep, model):
    receiver = copy.deepcopy(model)
    ev = sweep.SweepCommunicationEvaluator(Dataset(), Tokenizer(), False, 96)
    ids_a, ids_b = torch.tensor([[1, 4, 5]]), torch.tensor([[8, 9]])
    ev.prepare_input_ids = lambda *args: (ids_a, ids_b)
    with torch.no_grad():
        expected = model(ids_a, use_cache=True).past_key_values
    class Receiver:
        A, B = model, receiver

        def generate(self, ids, out_A_past_key_values, **kwargs):
            for actual, native in zip(out_A_past_key_values.layers, expected.layers):
                torch.testing.assert_close(actual.keys, native.keys)
                torch.testing.assert_close(actual.values, native.values)
            return torch.cat((ids, torch.tensor([[10]])), dim=-1)

    seen = []
    handle = model.register_forward_hook(lambda module, args, kwargs, out: seen.append(
        (kwargs.get("logits_to_keep"), out.logits.shape[-2])), with_kwargs=True)
    try:
        with torch.no_grad():
            assert ev.inference(model, Receiver(), {}) == "10"
    finally:
        handle.remove()
    assert seen == [(1, 1)]


def test_real_local_checkpoint_and_tokenizer_run_without_hub(sweep, model, tmp_path, monkeypatch):
    from tokenizers import Tokenizer as RawTokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    checkpoint = tmp_path / "checkpoint"
    model.save_pretrained(checkpoint)
    vocab = {str(i): i for i in range(1, 63)}
    vocab.update({"[PAD]": 0, "[EOS]": 63})
    raw = RawTokenizer(WordLevel(vocab, unk_token="1"))
    raw.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, unk_token="1",
                                        pad_token="[PAD]", eos_token="[EOS]")
    tokenizer.chat_template = "{{ messages[0]['content'] }}"
    tokenizer.save_pretrained(checkpoint)
    monkeypatch.setattr(sweep, "get_evaluator", lambda task: Dataset())
    result = sweep.main(config(sweep, tmp_path / "runs", model_A=str(checkpoint),
                               model_B=str(checkpoint)))
    assert len(result) == 2
    assert all(row["status"] == "success" for row in result)
    assert all(row["items_evaluated"] == 2 for row in result)


def test_oom_discards_partial_score_and_continues(sweep, offline, tmp_path, monkeypatch):
    native = sweep.SweepCommunicationEvaluator.inference
    calls = 0
    allocation = None

    def fail_second_example(self, model, cv, item):
        nonlocal calls, allocation
        if not cv.apply_attn_tracer:
            calls += 1
            if calls == 4:
                assert len(csv_rows(tmp_path)) == 1, "completed score was not saved immediately"
                temporary = torch.ones(16)
                allocation = weakref.ref(temporary)
                raise torch.OutOfMemoryError("simulated OOM")
            if calls == 5:
                assert allocation() is None, "failed inference traceback retained a tensor"
        return native(self, model, cv, item)

    monkeypatch.setattr(sweep.SweepCommunicationEvaluator, "inference", fail_second_example)
    result = sweep.main(config(sweep, tmp_path, alphas=[1.0, 0.5, 0.0]))
    assert [row["status"] for row in result] == ["success", "oom", "success"]
    assert result[1]["communication_result"] is None
    assert result[1]["items_evaluated"] == 1
    assert all(row["items_evaluated"] == 2 for row in (result[0], result[2]))
    assert [row["alpha"] for row in csv_rows(tmp_path)] == ["1.0", "0.0"]
    assert result[1]["sample_index"] == 2


@pytest.mark.parametrize("error_class", [torch.OutOfMemoryError, ValueError])
def test_calibration_failure_skips_alphas_then_continues_dataset(sweep, offline, tmp_path, monkeypatch, error_class):
    native = sweep.SweepCommunicationEvaluator.inference

    def fail_calibration(self, *args):
        if self.evaluator.name == "tipsheets":
            raise (torch.OutOfMemoryError if error_class is torch.OutOfMemoryError
                   else sweep.InputLimitError)("context too long")
        return native(self, *args)

    monkeypatch.setattr(sweep.SweepCommunicationEvaluator, "inference", fail_calibration)
    result = sweep.main(config(sweep, tmp_path, test_tasks=["tipsheets", "countries"]))
    assert [row["status"] for row in result] == ["calibration_failed", "calibration_failed", "success", "success"]
    assert len(csv_rows(tmp_path)) == 2
    assert all(row["test_task"] == "countries" for row in csv_rows(tmp_path))


def test_existing_results_append_without_duplicate_header(sweep, offline, tmp_path):
    output = tmp_path / "shared.csv"
    sweep.main(config(sweep, tmp_path / "one", results_csv=str(output), alphas=[1.0]))
    sweep.main(config(sweep, tmp_path / "two", results_csv=str(output), alphas=[0.0]))
    assert [row["alpha"] for row in csv_rows(tmp_path)] == ["1.0", "0.0"]


def test_csv_and_journal_cannot_share_a_file(sweep, offline, tmp_path):
    with pytest.raises(ValueError, match="distinct"):
        sweep.main(config(sweep, tmp_path, results_csv=str(tmp_path / "results.jsonl")))
    assert not offline.loaded


def test_wandb_runs_close_after_recoverable_failure(sweep, offline, tmp_path, monkeypatch):
    native = sweep.SweepCommunicationEvaluator.inference
    runs = []
    calls = 0

    def init(**kwargs):
        run = SimpleNamespace(closed=False, records=[])
        run.log = lambda record: run.records.append(record)
        run.finish = lambda: setattr(run, "closed", True)
        runs.append(run)
        return run

    def fail_first_case(self, model, cv, item):
        nonlocal calls
        if not cv.apply_attn_tracer:
            calls += 1
            if calls == 1:
                raise torch.OutOfMemoryError("simulated OOM")
        return native(self, model, cv, item)

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=init))
    monkeypatch.setattr(sweep.SweepCommunicationEvaluator, "inference", fail_first_case)
    result = sweep.main(config(sweep, tmp_path, use_wandb=True, wandb_project="offline-test"))
    assert [row["status"] for row in result] == ["oom", "success"]
    assert len(runs) == 2
    assert all(run.closed for run in runs)
    assert runs[0].records[0]["communication_result"] is None


def test_communicator_restores_existing_attention_attributes(sweep, offline, tmp_path):
    cfg = config(sweep, tmp_path)
    with sweep.ExperimentSession(cfg) as session:
        attention = session.model_B.model.layers[0].self_attn
        attention._kvcomm_trace = False
        attention._kvcomm_prefill = False
        attention.attn_inputs = None
        with pytest.raises(torch.OutOfMemoryError):
            with session.communicator(trace=True) as cv:
                attention.attn_inputs = (torch.ones(2),)
                cv.B_attn_weights[0] = torch.ones(2)
                raise torch.OutOfMemoryError("in context")
        assert cv.B_attn_weights == {}
        assert attention.attn_inputs is None
        assert not attention._kvcomm_trace
        assert not attention._kvcomm_prefill
        assert session.model_B.config._attn_implementation == "sdpa"


def test_unexpected_failure_is_recorded_then_raised(sweep, offline, tmp_path, monkeypatch):
    def fail(*args):
        raise RuntimeError("unexpected failure")

    monkeypatch.setattr(sweep.SweepCommunicationEvaluator, "inference", fail)
    with pytest.raises(RuntimeError, match="unexpected failure"):
        sweep.main(config(sweep, tmp_path))
    assert records(tmp_path)[-1]["status"] == "error"
    assert offline.loaded[1].config._attn_implementation == "sdpa"


def test_calibration_tensors_released_between_examples(sweep, offline, tmp_path, monkeypatch):
    native = sweep.SweepCommunicationEvaluator.inference
    calls = 0

    def check_cleared(self, model, cv, item):
        nonlocal calls
        if cv.apply_attn_tracer:
            calls += 1
            if calls == 2:
                assert cv.B_attn_weights == {}
                assert all(block.self_attn.attn_inputs is None for block in cv.B.model.layers)
        return native(self, model, cv, item)

    monkeypatch.setattr(sweep.SweepCommunicationEvaluator, "inference", check_cleared)
    sweep.main(config(sweep, tmp_path, calib_size=2))
    assert calls == 2

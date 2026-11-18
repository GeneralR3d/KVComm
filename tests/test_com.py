"""Offline CPU checks: python -m pytest -q tests/test_com.py."""

import copy
import csv
from fractions import Fraction
import importlib
import json
from types import SimpleNamespace

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM


@pytest.fixture
def model():
    torch.set_num_threads(1)
    torch.manual_seed(7)
    cfg = LlamaConfig(
        vocab_size=64, hidden_size=32, intermediate_size=48,
        num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, max_position_embeddings=128, attention_dropout=0.0,
        pad_token_id=0, bos_token_id=1, eos_token_id=63,
    )
    cfg._attn_implementation = "sdpa"
    return LlamaForCausalLM(cfg).eval()


class Tokenizer:
    eos_token_id = 63
    pad_token_id = 0

    def apply_chat_template(self, messages, **kwargs):
        words = messages[0]["content"].split()
        return torch.tensor([[1] + [2 + sum(map(ord, word)) % 50 for word in words]])

    def decode(self, ids, **kwargs):
        return " ".join(map(str, ids.tolist()))


@pytest.fixture
def com(monkeypatch):
    import nltk
    monkeypatch.setattr(nltk, "download", lambda *args, **kwargs: True)
    return importlib.import_module("com")


@pytest.fixture
def offline(com, model, monkeypatch):
    from dataloader.base_evaluator import BaseEvaluator
    import eval as evaluation

    class Dataset(BaseEvaluator):
        def __init__(self):
            super().__init__()
            self.name = "tipsheets"
            self.max_tokens = 2
            self.truncate_input = True
            self.multiple_answers = False
            self.data = [
                {"id": "first", "prompt_A": "a b c", "prompt_B": "q",
                 "answer": ["purple", "blue green"]},
                {"id": "second", "prompt_A": "d e f", "prompt_B": "r",
                 "answer": "blue green"},
            ]

    dataset, loaded, seen, traced = Dataset(), [], [], []

    def load(*args, **kwargs):
        value = copy.deepcopy(model)
        loaded.append(value)
        return value

    def importance(weights, model_name, total):
        for layer, score in enumerate([0, 0, 1]):
            total[layer].append(score)
        return total

    def respond(self, sender, cv, item):
        if cv.apply_attn_tracer:
            traced.append(item["id"])
            return "10"
        seen.append((tuple(cv.layers_list), item["id"]))
        if list(cv.layers_list) == [0, 1]:
            return "blue green red yellow cyan"
        return "blue green red" if item["id"] == "first" else "blue green"

    native = SimpleNamespace(inference=evaluation.CommunicationEvaluator.inference,
                             importance=evaluation.calc_layer_importance,
                             weights=com.CVCommunicator.calc_attn_weights_from_qk)
    monkeypatch.setattr(com.AutoModelForCausalLM, "from_pretrained", load)
    monkeypatch.setattr(com.AutoTokenizer, "from_pretrained", lambda *a, **k: Tokenizer())
    monkeypatch.setattr(com, "get_evaluator", lambda task: dataset)
    monkeypatch.setattr(com, "log_gpu_info", lambda: None)
    monkeypatch.setattr(com, "setup_logging", lambda **kwargs: None)
    monkeypatch.setattr("utils.f1.lemmatize_text", lambda value: value)
    monkeypatch.setattr(evaluation, "calc_layer_importance", importance)
    monkeypatch.setattr(com.CVCommunicator, "calc_attn_weights_from_qk", lambda self: None)
    monkeypatch.setattr(evaluation.CommunicationEvaluator, "inference", respond)
    return SimpleNamespace(dataset=dataset, loaded=loaded, seen=seen, traced=traced, native=native)


def config(com, tmp_path, **overrides):
    values = dict(device="cpu", snapshot_path=str(tmp_path), do_test=True,
                  top_layers=0.7, mu=0, sigma=1, calib_size=1)
    values.update(overrides)
    return com.AlignConfig(**values)


def test_breakpoint_calibration_records_each_set_and_retains_best_f1(com, offline, tmp_path):
    cfg = config(com, tmp_path, breakpoint_calibration=True, calib_size=2)
    result = com.main(cfg)
    assert offline.seen == [((0, 1), "first"), ((0, 1), "second"),
                            ((0, 2), "first"), ((0, 2), "second"),
                            ((0, 2), "first"), ((0, 2), "second")]
    assert result == 1.0  # Final run returns the task metric, separately from calibration F1.
    assert cfg.layers_list == [0, 2]
    with next(tmp_path.rglob("calibration_candidates.csv")).open() as output:
        rows = list(csv.DictReader(output))
    assert [json.loads(row["selected_layers"]) for row in rows] == [[0, 1], [0, 2]]
    assert [float(row["f1"]) for row in rows] == pytest.approx([4 / 7, 0.9])
    assert all(float(Fraction(row["alpha_exact"])) == float(row["alpha"]) for row in rows)
    winner = json.loads(next(tmp_path.rglob("calibration_selection.json")).read_text())
    assert winner["selected_layers"] == [0, 2]
    assert winner["f1"] == pytest.approx(0.9)
    assert winner["candidate_count"] == 2
    from transformers.models.llama.modeling_llama import LlamaAttention
    assert all(type(block.self_attn) is LlamaAttention for block in offline.loaded[1].model.layers)


@pytest.mark.parametrize("always_include_layer_0, expected", [(True, [0, 2]), (False, [2, 1])])
def test_disabled_calibration_uses_fixed_alpha_and_task_metric(
        com, offline, tmp_path, always_include_layer_0, expected):
    cfg = config(com, tmp_path, always_include_layer_0=always_include_layer_0)
    assert com.main(cfg) == 1.0
    assert list(cfg.layers_list) == expected
    assert len(offline.seen) == 2
    assert not list(tmp_path.rglob("calibration_candidates.csv"))


def test_cli_exposes_calibration_and_layer_zero_switches(com):
    cfg = com.parse_args(["--do_test", "--top_layers", "0.3", "--breakpoint_calibration",
                          "--no_always_include_layer_0"])
    assert cfg.breakpoint_calibration is True
    assert cfg.always_include_layer_0 is False
    assert com.parse_args([]).breakpoint_calibration is False
    assert com.parse_args([]).always_include_layer_0 is True


@pytest.mark.parametrize("override", [
    {"top_layers": 0}, {"top_layers": 0.01}, {"top_layers": 1.1},
    {"random_selection": True}, {"do_layer_curve": True}, {"do_test": False},
    {"calib_size": 0}, {"sigma": 0}, {"limit": -1},
])
def test_invalid_breakpoint_settings_fail_without_a_winner(com, offline, tmp_path, override):
    cfg = config(com, tmp_path, breakpoint_calibration=True, **override)
    with pytest.raises(ValueError):
        com.main(cfg)
    assert not offline.seen
    assert not list(tmp_path.rglob("calibration_selection.json"))


def test_candidate_failure_keeps_completed_scores_without_selecting_winner(
        com, offline, tmp_path, monkeypatch):
    import eval as evaluation
    original = evaluation.CommunicationEvaluator.inference

    def fail(self, sender, cv, item):
        if not cv.apply_attn_tracer and list(cv.layers_list) == [0, 2]:
            raise torch.OutOfMemoryError("candidate failed")
        return original(self, sender, cv, item)

    monkeypatch.setattr(evaluation.CommunicationEvaluator, "inference", fail)
    with pytest.raises(torch.OutOfMemoryError):
        com.main(config(com, tmp_path, breakpoint_calibration=True))
    with next(tmp_path.rglob("calibration_candidates.csv")).open() as output:
        assert len(list(csv.DictReader(output))) == 1
    assert not list(tmp_path.rglob("calibration_selection.json"))


def test_repeated_runs_preserve_scores_and_do_not_reuse_a_stale_winner(
        com, offline, tmp_path, monkeypatch):
    import eval as evaluation
    cfg = config(com, tmp_path, breakpoint_calibration=True, run_name="repeat")
    com.main(cfg)

    def fail(*args, **kwargs):
        raise torch.OutOfMemoryError("second run failed")

    monkeypatch.setattr(evaluation.CommunicationEvaluator, "inference", fail)
    with pytest.raises(torch.OutOfMemoryError):
        com.main(config(com, tmp_path, breakpoint_calibration=True, run_name="repeat"))
    assert len(list(tmp_path.glob("repeat_*"))) == 2
    assert len(list(tmp_path.rglob("calibration_selection.json"))) == 1
    with next(tmp_path.rglob("calibration_candidates.csv")).open() as output:
        assert len(list(csv.DictReader(output))) == 2


def test_unconstrained_breakpoint_candidates_can_exclude_layer_zero(com):
    from layer_importance import enumerate_layer_candidates
    candidates = enumerate_layer_candidates([0, 0, 1], [0, 1, 0], 1,
                                            always_include_layer_0=False)
    assert candidates == [(Fraction(1, 4), (1,)), (Fraction(3, 4), (2,))]


def test_breakpoint_candidates_include_a_narrow_interval(com):
    middle = 0.5 + 2**-45
    candidates = com.enumerate_layer_candidates([0, 0, middle, 1], [0, 1, middle, 0], 2)
    assert [layers for _, layers in candidates] == [(0, 1), (0, 2), (0, 3)]


def test_breakpoint_midpoints_stay_distinct_when_floats_round_to_one(com):
    tiny = 2**-60
    candidates = {layers: alpha for alpha, layers in
                  com.enumerate_layer_candidates([1, 0, tiny, 2 * tiny], [0, 1, 0.75, 0], 2)}
    assert candidates[(0, 2)] < candidates[(0, 3)] < 1
    assert float(candidates[(0, 2)]) == float(candidates[(0, 3)]) == 1.0


def test_breakpoint_candidates_deduplicate_sets_and_use_stable_ties(com):
    candidates = com.enumerate_layer_candidates([0, 0, 1, 0], [0, 1, 0, 0], 3)
    assert candidates == [(Fraction(1, 4), (0, 1, 2))]
    assert com.enumerate_layer_candidates([0, 1, 1], [0, 1, 1], 2) == [(Fraction(1, 2), (0, 1))]
    assert com.enumerate_layer_candidates([0, 0, 1], [0, 1, 0], 1) == [(Fraction(1, 2), (0,))]


def test_random_selection_reserves_one_layer_zero_slot(com, offline, tmp_path):
    cfg = config(com, tmp_path, random_selection=True)
    com.main(cfg)
    assert len(cfg.layers_list) == 2
    assert 0 in cfg.layers_list


@pytest.mark.parametrize("calib_size, limit, candidate_ids, final_ids, expected_f1", [
    (1, 2, ["first"], ["first", "second"], 0.8),
    (2, 1, ["first", "second"], ["first"], 0.9),
    (1, 0, ["first"], ["first", "second"], 0.8),
    (5, 1, ["first", "second"], ["first"], 0.9),
])
def test_calib_size_controls_both_calibration_stages_and_limit_only_controls_final_run(
        com, offline, tmp_path, calib_size, limit, candidate_ids, final_ids, expected_f1):
    cfg = config(com, tmp_path, breakpoint_calibration=True, calib_size=calib_size, limit=limit)
    assert com.main(cfg) == 1.0
    assert offline.traced == candidate_ids
    assert offline.seen == ([((0, 1), item) for item in candidate_ids]
                            + [((0, 2), item) for item in candidate_ids]
                            + [((0, 2), item) for item in final_ids])
    winner = json.loads(next(tmp_path.rglob("calibration_selection.json")).read_text())
    assert winner["f1"] == pytest.approx(expected_f1)


@pytest.mark.parametrize("calib_size, expected_layers, expected_result", [
    (1, [0, 1], 0.5), (2, [0, 2], 1.0),
])
def test_winner_is_selected_using_calibration_examples_before_full_run(
        com, offline, tmp_path, monkeypatch, calib_size, expected_layers, expected_result):
    import eval as evaluation

    def respond(self, sender, cv, item):
        if cv.apply_attn_tracer:
            return "10"
        if list(cv.layers_list) == [0, 1]:
            return "blue green" if item["id"] == "first" else "unknown"
        return "blue green red" if item["id"] == "first" else "blue green"

    monkeypatch.setattr(evaluation.CommunicationEvaluator, "inference", respond)
    cfg = config(com, tmp_path, breakpoint_calibration=True, calib_size=calib_size, limit=2)
    assert com.main(cfg) == expected_result
    assert cfg.layers_list == expected_layers


def test_one_slot_budget_evaluates_only_layer_zero(com, offline, tmp_path):
    cfg = config(com, tmp_path, breakpoint_calibration=True, top_layers=0.34)
    com.main(cfg)
    assert cfg.layers_list == [0]
    assert offline.seen == [((0,), "first"), ((0,), "first"), ((0,), "second")]


def test_unconstrained_calibration_evaluates_sets_without_reserving_zero(com, offline, tmp_path):
    cfg = config(com, tmp_path, breakpoint_calibration=True, top_layers=0.34,
                 always_include_layer_0=False, mu=0.5)
    com.main(cfg)
    assert offline.seen == [((1,), "first"), ((2,), "first"),
                            ((1,), "first"), ((1,), "second")]
    assert cfg.layers_list == [1]  # The first interval wins an F1 tie.


def test_receiver_attention_is_mapped_to_sender_layer_ids(com, offline, tmp_path, monkeypatch):
    import eval as evaluation
    original_load = com.AutoModelForCausalLM.from_pretrained

    def load(*args, **kwargs):
        value = original_load(*args, **kwargs)
        if len(offline.loaded) == 2:
            value.model.layers = value.model.layers[:2]
            value.config.num_hidden_layers = 2
        return value

    def importance(weights, model_name, total):
        total[0].append(0)
        total[1].append(1)
        return total

    monkeypatch.setattr(com.AutoModelForCausalLM, "from_pretrained", load)
    monkeypatch.setattr(evaluation, "calc_layer_importance", importance)
    cfg = config(com, tmp_path, breakpoint_calibration=True)
    com.main(cfg)
    assert cfg.layers_list == [0, 2]


def test_real_cpu_tracing_releases_tensors_and_restores_attention(com, offline, tmp_path, monkeypatch):
    import eval as evaluation
    from transformers.models.llama.modeling_llama import LlamaAttention

    trace_calls, sender_logits = [], []

    def infer(self, sender, cv, item):
        if cv.apply_attn_tracer:
            assert cv.B_attn_weights == {}
            assert all(block.self_attn.attn_inputs is None for block in cv.B.model.layers)
            trace_calls.append(item["id"])
        handle = sender.register_forward_hook(
            lambda module, args, kwargs, out: sender_logits.append(out.logits.shape[-2]),
            with_kwargs=True)
        try:
            return offline.native.inference(self, sender, cv, item)
        finally:
            handle.remove()

    monkeypatch.setattr(evaluation.CommunicationEvaluator, "inference", infer)
    monkeypatch.setattr(evaluation, "calc_layer_importance", offline.native.importance)
    monkeypatch.setattr(com.CVCommunicator, "calc_attn_weights_from_qk", offline.native.weights)
    cfg = config(com, tmp_path, breakpoint_calibration=True, calib_size=2)
    assert com.main(cfg) == 0.0
    assert trace_calls == ["first", "second"]
    assert all(value == 1 for value in sender_logits)
    assert all(type(block.self_attn) is LlamaAttention for block in offline.loaded[1].model.layers)

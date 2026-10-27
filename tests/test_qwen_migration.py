"""CPU integration tests: tiny random models, no Hub access or GPU required."""

import copy

import pytest
import torch
from transformers import (
    Qwen2Config, Qwen2ForCausalLM,
    Qwen3Config, Qwen3ForCausalLM,
    Qwen3MoeConfig, Qwen3MoeForCausalLM,
)


torch.set_num_threads(1)


@pytest.fixture(params=["qwen2", "qwen3", "qwen3_moe"])
def model(request):
    torch.manual_seed(7)
    config_class, model_class = {
        "qwen2": (Qwen2Config, Qwen2ForCausalLM),
        "qwen3": (Qwen3Config, Qwen3ForCausalLM),
        "qwen3_moe": (Qwen3MoeConfig, Qwen3MoeForCausalLM),
    }[request.param]
    config = config_class(
        vocab_size=64, hidden_size=32, intermediate_size=48,
        num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, max_position_embeddings=128, attention_dropout=0.0,
        num_experts=4, num_experts_per_tok=2, moe_intermediate_size=16,
        pad_token_id=0, bos_token_id=1, eos_token_id=63,
    )
    config._attn_implementation = "sdpa"
    return model_class(config).eval()


def communicator(model, **kwargs):
    from models_qwen import CVCommunicator
    return CVCommunicator(model, copy.deepcopy(model), 0, 2, **kwargs).eval()


@torch.no_grad()
def test_selected_cache_preserves_sink_without_mutating_donor(model):
    cv = communicator(model, layers_list=[2])
    donor = model(torch.tensor([[1, 4, 5, 6]]), use_cache=True).past_key_values
    original = [layer.keys.clone() for layer in donor.layers]
    selected = cv.prepare_key_cache(donor)
    assert [layer.keys.shape[-2] for layer in selected.layers] == [4, 1, 4]
    for i in range(3):
        expected = original[i] if i in (0, 2) else original[i][..., :1, :]
        torch.testing.assert_close(selected.layers[i].keys, expected)
    cv(torch.tensor([[8, 9]]), out_A_past_key_values=donor)
    for i in range(3):
        torch.testing.assert_close(donor.layers[i].keys, original[i])


@torch.no_grad()
def test_all_layers_match_native_concatenated_prompt(model):
    cv = communicator(model, layers_list=[0, 1, 2])
    donor_ids = torch.tensor([[1, 4, 5, 6]])
    receiver_ids = torch.tensor([[8, 9, 10]])
    donor = model(donor_ids, use_cache=True).past_key_values
    expected = cv.B(torch.cat((donor_ids, receiver_ids), dim=-1)).logits[:, -3:]
    actual = cv(receiver_ids, attention_mask=torch.ones_like(receiver_ids),
                out_A_past_key_values=donor).logits
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


@torch.no_grad()
def test_generate_uses_donor_and_matches_native_continuation(model):
    cv = communicator(model, layers_list=[0, 1, 2])
    donor_ids = torch.tensor([[1, 4, 5, 6]])
    receiver_ids = torch.tensor([[8, 9]])
    donor = model(donor_ids, use_cache=True).past_key_values
    expected = cv.B.generate(torch.cat((donor_ids, receiver_ids), -1),
                             max_new_tokens=3, do_sample=False)
    actual = cv.generate(receiver_ids, attention_mask=torch.ones_like(receiver_ids),
                         out_A_past_key_values=donor,
                         max_new_tokens=3, do_sample=False)
    assert actual.shape[-1] == receiver_ids.shape[-1] + 3
    torch.testing.assert_close(actual[:, -3:], expected[:, -3:])


@torch.no_grad()
def test_sink_layer_attention_is_causal_and_decode_grows_each_layer(model):
    cv = communicator(model, layers_list=[2], apply_attn_tracer=True)
    donor = model(torch.tensor([[1, 4, 5, 6]]), use_cache=True).past_key_values
    output = cv(torch.tensor([[8, 9, 10]]), out_A_past_key_values=donor)
    cv.calc_attn_weights_from_qk()
    weights = cv.B_attn_weights[1]
    assert weights.shape[-2:] == (3, 4)
    assert torch.count_nonzero(weights[..., 0, 2:]) == 0
    assert torch.count_nonzero(weights[..., 1, 3:]) == 0
    torch.testing.assert_close(weights.sum(-1), torch.ones_like(weights.sum(-1)))
    decoded = cv(torch.tensor([[11]]), past_key_values=output.past_key_values,
                 out_A_past_key_values=donor)
    assert [layer.keys.shape[-2] for layer in decoded.past_key_values.layers] == [8, 5, 8]


@torch.no_grad()
def test_tracing_preserves_native_logits_and_captures_all_moe_layers(model):
    native = copy.deepcopy(model)
    cv = communicator(model, apply_attn_tracer=True)
    ids = torch.tensor([[1, 4, 5]])
    expected = native(ids).logits
    actual = cv.B(ids).logits
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    cv.calc_attn_weights_from_qk()
    assert set(cv.B_attn_weights) == {0, 1, 2}
    for weights in cv.B_attn_weights.values():
        assert weights.shape == (1, 4, 3, 3)


@torch.no_grad()
def test_multi_sender_concatenates_only_selected_layers(model):
    from models_qwen import MultiSenderCVCommunicator
    cv = MultiSenderCVCommunicator(model, copy.deepcopy(model), copy.deepcopy(model),
                                    0, 2, layers_list=[2], apply_attn_tracer=True)
    first = model(torch.tensor([[1, 4, 5]]), use_cache=True).past_key_values
    second = model(torch.tensor([[1, 6]]), use_cache=True).past_key_values
    cache = cv.prepare_key_cache(first, second)
    assert [layer.keys.shape[-2] for layer in cache.layers] == [5, 1, 5]
    torch.testing.assert_close(cache.layers[2].keys,
                               torch.cat((first.layers[2].keys, second.layers[2].keys), -2))
    output = cv.generate(torch.tensor([[8, 9]]), out_A1_past_key_values=first,
                         out_A2_past_key_values=second, max_new_tokens=2)
    assert output.shape == (1, 4)
    cv.calc_attn_weights_from_qk()
    assert cv.B_attn_weights[0].shape[-2:] == (2, 7)


@pytest.mark.parametrize("kwargs", [
    {"layers_list": []}, {"layers_list": [-2]}, {"layers_list": [3]},
    {"layer_from": 2, "layer_to": 1}, {"shift_back": True},
])
def test_invalid_transfer_options_fail_clearly(model, kwargs):
    from models_qwen import CVCommunicator
    options = dict(layer_from=0, layer_to=2)
    options.update(kwargs)
    with pytest.raises(ValueError):
        CVCommunicator(model, copy.deepcopy(model), **options)


def test_incompatible_head_geometry_is_rejected(model):
    from models_qwen import CVCommunicator
    other = copy.deepcopy(model)
    other.config.num_key_value_heads = 1
    with pytest.raises(ValueError, match="KV"):
        CVCommunicator(model, other, 0, 2)


@torch.no_grad()
def test_single_token_receiver_prompt_still_initializes_transfer(model):
    cv = communicator(model)
    donor = model(torch.tensor([[1, 4, 5]]), use_cache=True).past_key_values
    output = cv(torch.tensor([[8]]), out_A_past_key_values=donor)
    assert output.past_key_values.get_seq_length() == 4


@pytest.mark.parametrize("operation", ["replace", "sum", "mean"])
@torch.no_grad()
def test_activation_injection_matches_native_layer_hook(model, operation):
    from models_qwen import ActivationCommunicator
    native = copy.deepcopy(model)
    receiver = copy.deepcopy(model)
    hidden = model(torch.tensor([[1, 4, 5]]), output_hidden_states=True).hidden_states
    cv = ActivationCommunicator(model, receiver, layer_k=1, layer_j=1, f=operation)
    ids = torch.tensor([[8, 9]])
    def inject(module, args):
        states = args[0].clone()
        a, b = hidden[1][:, -1], states[:, -1]
        states[:, -1] = {"replace": a, "sum": a + b, "mean": (a + b) / 2}[operation]
        return (states,) + args[1:]
    handle = native.model.layers[1].register_forward_pre_hook(inject)
    try:
        expected = native(ids).logits
    finally:
        handle.remove()
    actual = cv(ids, h_A=hidden).logits
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    # The temporary hook must not affect subsequent native calls or generation.
    torch.testing.assert_close(receiver(ids).logits, model(ids).logits)
    assert cv.generate(ids, h_A=hidden, max_new_tokens=2).shape == (1, 4)


@torch.no_grad()
def test_repeated_calibration_uses_current_prompt_and_can_be_disabled(model):
    cv = communicator(model, apply_attn_tracer=True)
    for length in (4, 2):
        donor = model(torch.tensor([[1, 4, 5]]), use_cache=True).past_key_values
        cv(torch.tensor([[8] * length]), out_A_past_key_values=donor)
        cv.calc_attn_weights_from_qk()
        assert cv.B_attn_weights[0].shape[-2] == length
    from models_qwen import CVCommunicator
    plain = CVCommunicator(model, cv.B, 0, 2, apply_attn_tracer=False)
    donor = model(torch.tensor([[1, 4, 5]]), use_cache=True).past_key_values
    plain(torch.tensor([[8, 9]]), out_A_past_key_values=donor)
    assert all(block.self_attn.attn_inputs is None for block in plain.B.model.layers)


@torch.no_grad()
def test_migrated_loader_reads_local_dense_and_moe_checkpoints(model, tmp_path):
    from model_runtime import load_experiment_model
    model.save_pretrained(tmp_path)
    loaded = load_experiment_model(str(tmp_path), "cpu", torch.float32).eval()
    ids = torch.tensor([[1, 4, 5]])
    torch.testing.assert_close(loaded(ids).logits, model(ids).logits)


def test_sliding_attention_is_rejected(model):
    from models_qwen import CVCommunicator
    other = copy.deepcopy(model)
    other.config.layer_types = ["sliding_attention"] * 3
    with pytest.raises(ValueError, match="full-attention"):
        CVCommunicator(model, other)

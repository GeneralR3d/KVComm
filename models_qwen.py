"""Qwen-only KV communication for Transformers 5.21.0.

Keep upstream attention projections, Q/K normalization, RoPE, decoder layers,
and MoE routing intact. The custom SDPA backend only aligns selective-cache
masks and captures the actual attention inputs during calibration.
"""

import copy

import torch
from transformers import PreTrainedModel
from transformers.cache_utils import DynamicCache, DynamicLayer
from transformers.generation import GenerationMixin
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.masking_utils import AttentionMaskInterface, eager_mask
from transformers.modeling_utils import AttentionInterface


QWEN_MODEL_TYPES = {"qwen2", "qwen3", "qwen3_moe"}
_BACKEND = "kvcomm_qwen_sdpa"


def _attention_forward(module, query, key, value, attention_mask,
                       scaling=None, dropout=0.0, **kwargs):
    query_length, key_length = query.shape[-2], key.shape[-2]
    if attention_mask is not None and attention_mask.shape[-1] != key_length:
        if attention_mask.shape[-1] < key_length:
            raise ValueError("Attention mask is shorter than the Qwen KV cache")
        # A pruned layer retains the first donor token and all receiver tokens.
        # Taking the first key_length columns would unmask future receiver tokens.
        tail = attention_mask[..., -(key_length - 1):] if key_length > 1 else attention_mask[..., :0]
        attention_mask = torch.cat((attention_mask[..., :1], tail), dim=-1)
    if attention_mask is None and key_length > query_length and query_length > 1:
        q = torch.arange(query_length, device=query.device) + key_length - query_length
        k = torch.arange(key_length, device=query.device)
        attention_mask = (k[None, :] <= q[:, None])[None, None]

    if getattr(module, "_kvcomm_trace", False) and (
        query_length > 1 or getattr(module, "_kvcomm_prefill", False)
    ):
        module.attn_inputs = (
            query.detach(), key.detach(),
            attention_mask.detach() if attention_mask is not None else None,
            scaling,
        )
    return sdpa_attention_forward(
        module, query, key, value, attention_mask,
        scaling=scaling, dropout=dropout, **kwargs,
    )


AttentionInterface.register(_BACKEND, _attention_forward)
# Always retain a concrete mask: cache lengths differ between full and sink layers.
AttentionMaskInterface.register(_BACKEND, eager_mask)


def _validate_models(models):
    for model in models:
        config = model.config
        if config.model_type not in QWEN_MODEL_TYPES:
            raise ValueError(f"Unsupported Qwen architecture: {config.model_type}")
        if any(kind != "full_attention" for kind in getattr(config, "layer_types", [])):
            raise ValueError("Qwen KV transfer currently requires full-attention layers")
        if config.model_type == "qwen3_moe" and config.sliding_window is not None:
            raise ValueError("Qwen KV transfer does not support sliding-window caches")
    receiver = models[-1].config
    for model in models[:-1]:
        config = model.config
        if config.model_type != receiver.model_type:
            raise ValueError("Qwen senders and receiver must use the same architecture")
        if config.num_hidden_layers != receiver.num_hidden_layers:
            raise ValueError("Qwen senders and receiver must have equal layer counts")
        def geometry(cfg):
            return (cfg.num_key_value_heads,
                    getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads)
        if geometry(config) != geometry(receiver):
            raise ValueError("Qwen sender and receiver KV head geometry must match")


class CVCommunicator(PreTrainedModel, GenerationMixin):
    _supports_sdpa = True
    _supports_attention_backend = True

    def __init__(self, model_A, model_B, layer_from=0, layer_to=None,
                 top_layers=0.0, layers_list=None, apply_attn_tracer=False,
                 shift_back=False):
        if shift_back:
            raise ValueError("shift_back is outside the Qwen migration scope")
        _validate_models((model_A, model_B))
        wrapper_config = copy.deepcopy(model_B.config)
        # This wrapper has no experts of its own. B keeps its native expert
        # backend; do not ask PreTrainedModel to dispatch experts on the wrapper.
        wrapper_config._experts_implementation = "eager"
        super().__init__(wrapper_config)
        self.A, self.B = model_A, model_B
        self.A_num_layers = model_A.config.num_hidden_layers
        self.B_num_layers = model_B.config.num_hidden_layers
        self.layer_map = dict(enumerate(range(self.A_num_layers)))
        layer_to = self.A_num_layers - 1 if layer_to is None else layer_to
        if not 0 <= top_layers <= 1:
            raise ValueError("top_layers must be between 0 and 1")
        if layers_list is None or list(layers_list) == [-1]:
            if top_layers > 0:
                layers_list = list(range(self.A_num_layers))
            else:
                if not 0 <= layer_from <= layer_to:
                    raise ValueError("Invalid Qwen layer range")
                # The legacy CLI defaults layer_to to 26 even for smaller models.
                layers_list = list(range(layer_from, min(layer_to + 1, self.A_num_layers)))
        if len(layers_list) == 0 or any(i < 0 or i >= self.A_num_layers for i in layers_list):
            raise ValueError("layers_list must contain valid Qwen layer indices")
        self.layers_list = list(layers_list)
        self.apply_attn_tracer = apply_attn_tracer
        self.B_attn_weights = {}
        self.generation_config = copy.deepcopy(model_B.generation_config)
        self.B.set_attn_implementation(_BACKEND)
        for block in self.B.model.layers:
            block.self_attn._kvcomm_trace = apply_attn_tracer
            block.self_attn.attn_inputs = None
        for model in (self.A, self.B):
            model.requires_grad_(False)

    def get_input_embeddings(self):
        return self.B.get_input_embeddings()

    def _prepare_caches(self, donors):
        cache = DynamicCache(config=self.B.config)
        for donor in donors:
            if not isinstance(donor, DynamicCache) or len(donor.layers) != self.A_num_layers:
                raise ValueError("A complete Qwen DynamicCache is required for each sender")
            if any(type(layer) is not DynamicLayer for layer in donor.layers):
                raise ValueError("Only full-attention Qwen DynamicCache layers are supported")
        batch_size = None
        for i, block in enumerate(self.B.model.layers):
            parts = []
            for donor in donors:
                key, value = donor.layers[i].keys, donor.layers[i].values
                if key is None or value is None or key.ndim != 4 or key.shape != value.shape:
                    raise ValueError("Qwen donor KV tensors must be populated and have matching 4D shapes")
                if key.shape[1] != self.B.config.num_key_value_heads or key.shape[-1] != block.self_attn.head_dim:
                    raise ValueError("Qwen donor KV tensor dimensions do not match the receiver")
                if key.shape[-2] == 0 or (batch_size is not None and key.shape[0] != batch_size):
                    raise ValueError("Qwen donor caches must be nonempty and have equal batch sizes")
                batch_size = key.shape[0]
                parts.append((key, value))
            if i != 0 and i not in self.layers_list:
                key, value = parts[0]
                key, value = key[..., :1, :], value[..., :1, :]
            else:
                key = torch.cat([part[0] for part in parts], dim=-2)
                value = torch.cat([part[1] for part in parts], dim=-2)
            target = block.self_attn.k_proj.weight
            cache.update(key.to(device=target.device, dtype=target.dtype),
                         value.to(device=target.device, dtype=target.dtype), i)
        return cache

    def prepare_key_cache(self, past_key_values):
        return self._prepare_caches((past_key_values,))

    def prepare_inputs_for_generation(self, input_ids, next_sequence_length=None,
                                      past_key_values=None, attention_mask=None, **kwargs):
        # Generation positions cover B's visible prompt only. B derives the actual
        # RoPE positions from its transferred cache, shared across all layers.
        kwargs.pop("position_ids", None)
        kwargs.pop("is_first_iteration", None)
        if next_sequence_length is not None:
            input_ids = input_ids[:, -next_sequence_length:]
        return dict(input_ids=input_ids, past_key_values=past_key_values,
                    attention_mask=attention_mask, **kwargs)

    def _forward_receiver(self, input_ids, cache, attention_mask, prefill, **kwargs):
        if not isinstance(cache, DynamicCache):
            raise ValueError("Qwen communication requires a DynamicCache")
        if attention_mask is not None:
            if attention_mask.ndim != 2:
                raise ValueError("Pass a 2D receiver attention_mask to Qwen communication")
            total_length = cache.get_seq_length() + input_ids.shape[-1]
            prefix_length = total_length - attention_mask.shape[-1]
            if prefix_length < 0:
                raise ValueError("Receiver attention_mask is longer than its cache and input")
            attention_mask = torch.cat((attention_mask.new_ones((input_ids.shape[0], prefix_length)),
                                        attention_mask), dim=-1)
        for block in self.B.model.layers:
            block.self_attn._kvcomm_prefill = prefill
        if prefill:
            self.B_attn_weights = {}
            for block in self.B.model.layers:
                block.self_attn.attn_inputs = None
        return self.B(input_ids=input_ids, attention_mask=attention_mask,
                      past_key_values=cache, **kwargs)

    def forward(self, input_ids, attention_mask=None, past_key_values=None,
                out_A_past_key_values=None, **kwargs):
        prefill = past_key_values is None or past_key_values.get_seq_length() == 0
        if prefill:
            if out_A_past_key_values is None:
                raise ValueError("out_A_past_key_values is required for Qwen prefill")
            past_key_values = self.prepare_key_cache(out_A_past_key_values)
        return self._forward_receiver(input_ids, past_key_values, attention_mask, prefill, **kwargs)

    @torch.no_grad()
    def calc_attn_weights_from_qk(self):
        if not self.apply_attn_tracer:
            raise ValueError("apply_attn_tracer must be True for calibration")
        for i, block in enumerate(self.B.model.layers):
            inputs = block.self_attn.attn_inputs
            if inputs is None:
                raise ValueError("Run Qwen prefill before calculating attention weights")
            query, key, mask, scaling = inputs
            key = key.repeat_interleave(block.self_attn.num_key_value_groups, dim=1)
            scores = torch.matmul(query, key.transpose(-2, -1)) * scaling
            if mask is not None:
                if mask.dtype == torch.bool:
                    scores = scores.masked_fill(~mask, float("-inf"))
                else:
                    scores = scores + mask
            else:
                q_len, k_len = query.shape[-2], key.shape[-2]
                allowed = torch.arange(k_len, device=query.device)[None, :] <= (
                    torch.arange(q_len, device=query.device)[:, None] + k_len - q_len)
                scores = scores.masked_fill(~allowed, float("-inf"))
            self.B_attn_weights[i] = scores.softmax(-1, dtype=torch.float32).to(query.dtype)


class MultiSenderCVCommunicator(CVCommunicator):
    def __init__(self, model_A1, model_A2, model_B, layer_from=0, layer_to=None, **kwargs):
        _validate_models((model_A1, model_A2, model_B))
        super().__init__(model_A1, model_B, layer_from, layer_to, **kwargs)
        self.A1, self.A2 = model_A1, model_A2
        self.A1_num_layers = self.A_num_layers
        self.A2_num_layers = model_A2.config.num_hidden_layers
        self.A2.requires_grad_(False)

    def prepare_key_cache(self, A1_past_key_values, A2_past_key_values):
        return self._prepare_caches((A1_past_key_values, A2_past_key_values))

    def forward(self, input_ids, attention_mask=None, past_key_values=None,
                out_A1_past_key_values=None, out_A2_past_key_values=None, **kwargs):
        prefill = past_key_values is None or past_key_values.get_seq_length() == 0
        if prefill:
            if out_A1_past_key_values is None or out_A2_past_key_values is None:
                raise ValueError("Both sender caches are required for Qwen prefill")
            past_key_values = self.prepare_key_cache(out_A1_past_key_values, out_A2_past_key_values)
        return self._forward_receiver(input_ids, past_key_values, attention_mask, prefill, **kwargs)


class ActivationCommunicator(CVCommunicator):
    """Inject A's last-token activation using a temporary native-layer pre-hook."""

    def __init__(self, model_A, model_B, layer_k=-2, layer_j=-2, f="replace"):
        super().__init__(model_A, model_B)
        self.k = layer_k if layer_k >= 0 else self.A_num_layers + layer_k
        self.j = layer_j if layer_j >= 0 else self.B_num_layers + layer_j
        if not 0 <= self.k < self.A_num_layers or not 0 <= self.j < self.B_num_layers:
            raise ValueError("Activation layer indices are outside the Qwen model")
        if f not in {"replace", "sum", "mean"}:
            raise ValueError("Activation operation must be replace, sum, or mean")
        self.f = f

    def forward(self, input_ids, attention_mask=None, past_key_values=None, h_A=None, **kwargs):
        prefill = past_key_values is None or past_key_values.get_seq_length() == 0
        handle = None
        if prefill:
            if h_A is None:
                raise ValueError("h_A is required for Qwen activation prefill")
            activation = h_A[self.k][:, -1, :]

            def inject(module, args):
                states = args[0].clone()
                a = activation.to(device=states.device, dtype=states.dtype)
                if a.shape != states[:, -1].shape:
                    raise ValueError("Qwen activation batch and hidden dimensions must match")
                if self.f == "replace":
                    states[:, -1] = a
                elif self.f == "sum":
                    states[:, -1] = a + states[:, -1]
                else:
                    states[:, -1] = (a + states[:, -1]) / 2
                return (states,) + args[1:]

            handle = self.B.model.layers[self.j].register_forward_pre_hook(inject)
        try:
            return self.B(input_ids=input_ids, attention_mask=attention_mask,
                          past_key_values=past_key_values, **kwargs)
        finally:
            if handle is not None:
                handle.remove()

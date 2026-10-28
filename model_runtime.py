"""Select Qwen's migrated runtime without importing legacy model internals."""

from importlib import import_module, metadata


QWEN_MODEL_TYPES = {"qwen2", "qwen3", "qwen3_moe"}
QWEN_TRANSFORMERS_VERSION = "5.21.0"


def validate_qwen_config(config):
    model_type = config.model_type
    if model_type.startswith("qwen"):
        if model_type not in QWEN_MODEL_TYPES:
            raise ValueError(
                f"Unsupported Qwen architecture: {model_type}. This migration covers "
                "Qwen2/2.5, Qwen3, and Qwen3 MoE with full attention."
            )
        if metadata.version("transformers") != QWEN_TRANSFORMERS_VERSION:
            raise ValueError(
                f"The Qwen runtime requires transformers=={QWEN_TRANSFORMERS_VERSION}. "
                "Install requirements-qwen.txt in a separate environment."
            )
        return True
    return False


def _is_qwen_pair(models):
    flags = [validate_qwen_config(model.config) for model in models]
    if any(flags) and not all(flags):
        raise ValueError("Mixing Qwen with legacy Llama/Gemma models is not supported")
    return all(flags)


def get_cv_communicator(*models):
    if len(models) not in (2, 3):
        raise ValueError("Expected one or two senders followed by the receiver")
    if _is_qwen_pair(models):
        module = import_module("models_qwen")
        return module.CVCommunicator if len(models) == 2 else module.MultiSenderCVCommunicator
    return import_module("models" if len(models) == 2 else "models_ms").CVCommunicator


def get_activation_communicator(*models):
    module = import_module("models_qwen" if _is_qwen_pair(models) else "models_ac")
    return module.ActivationCommunicator


def load_experiment_model(model_name, device, dtype):
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(model_name)
    is_qwen = validate_qwen_config(config)
    dtype_kwargs = {"dtype" if is_qwen else "torch_dtype": dtype}
    return AutoModelForCausalLM.from_pretrained(
        model_name, config=config, device_map={"": device},
        attn_implementation="sdpa", **dtype_kwargs,
    )


class _QwenTokenizer:
    def __init__(self, tokenizer, enable_thinking):
        self.tokenizer = tokenizer
        self.enable_thinking = enable_thinking

    def __getattr__(self, name):
        return getattr(self.tokenizer, name)

    def __call__(self, *args, **kwargs):
        return self.tokenizer(*args, **kwargs)

    def apply_chat_template(self, *args, **kwargs):
        kwargs.setdefault("enable_thinking", self.enable_thinking)
        # v5 tokenizers default to returning a BatchEncoding; existing evaluators
        # explicitly expect a tensor and move it directly to the model's device.
        kwargs.setdefault("return_dict", False)
        return self.tokenizer.apply_chat_template(*args, **kwargs)


def configure_qwen_tokenizer(tokenizer, models, enable_thinking=False):
    if _is_qwen_pair(models):
        return _QwenTokenizer(tokenizer, enable_thinking)
    return tokenizer

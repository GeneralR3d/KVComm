"""Runtime and tokenizer integration without Hub downloads."""

import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast


def test_qwen_routing_does_not_import_legacy_model_modules():
    result = subprocess.run([sys.executable, "-c", '''
from types import SimpleNamespace
import sys
from model_runtime import get_cv_communicator, get_activation_communicator
model = SimpleNamespace(config=SimpleNamespace(model_type="qwen3_moe"))
assert get_cv_communicator(model, model).__module__ == "models_qwen"
assert get_cv_communicator(model, model, model).__name__ == "MultiSenderCVCommunicator"
assert get_activation_communicator(model, model).__module__ == "models_qwen"
assert not ({"models", "models_ms", "models_ac", "model_attn"} & sys.modules.keys())
'''], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("model_type", ["qwen3_5", "qwen3_5_moe", "qwen2_vl"])
def test_unsupported_qwen_architecture_fails_before_loading_weights(model_type):
    from model_runtime import validate_qwen_config
    with pytest.raises(ValueError, match="Unsupported Qwen"):
        validate_qwen_config(SimpleNamespace(model_type=model_type))


@pytest.mark.parametrize("thinking, expected", [(False, 3), (True, 4)])
def test_qwen_tokenizer_returns_tensor_and_sets_thinking_mode(thinking, expected):
    from model_runtime import configure_qwen_tokenizer
    raw = Tokenizer(WordLevel({"[UNK]": 0, "hello": 1, "answer": 3, "think": 4}, unk_token="[UNK]"))
    raw.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, unk_token="[UNK]")
    tokenizer.chat_template = "{{ messages[0]['content'] }} {% if enable_thinking %}think{% else %}answer{% endif %}"
    model = SimpleNamespace(config=SimpleNamespace(model_type="qwen3_moe"))
    adapted = configure_qwen_tokenizer(tokenizer, (model, model), enable_thinking=thinking)
    ids = adapted.apply_chat_template([{"role": "user", "content": "hello"}],
                                      add_generation_prompt=True, return_tensors="pt")
    assert isinstance(ids, torch.Tensor)
    assert ids.tolist() == [[1, expected]]
    assert adapted.decode([1]) == "hello"


def test_legacy_tokenizer_object_is_returned_unchanged():
    from model_runtime import configure_qwen_tokenizer
    tokenizer = object()
    model = SimpleNamespace(config=SimpleNamespace(model_type="llama"))
    assert configure_qwen_tokenizer(tokenizer, (model, model)) is tokenizer


@pytest.mark.parametrize("entrypoint", ["com.py", "com_ms.py", "com_online.py"])
def test_entrypoint_help_imports_without_legacy_internals(entrypoint):
    # NLTK's module-level corpus downloads are unrelated external I/O. Suppress
    # only that operation while importing the actual CLI and all other modules.
    result = subprocess.run([sys.executable, "-c", '''
import nltk, runpy, sys
nltk.download = lambda *args, **kwargs: True
sys.argv = [sys.argv[1], "--help"]
try:
    runpy.run_path(sys.argv[0], run_name="__main__")
except SystemExit as exc:
    assert exc.code == 0
assert not ({"models", "models_ms", "models_ac", "model_attn"} & sys.modules.keys())
''', entrypoint], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "--qwen_enable_thinking" in result.stdout

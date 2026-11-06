from __future__ import annotations

from pathlib import Path

import pytest

from engine.config import ModelConfig

DATA = Path(__file__).parent / "data"
CONFIG_1B = DATA / "configs" / "llama-3.2-1b"
CONFIG_8B = DATA / "configs" / "llama-3.1-8b"

# A tiny Llama with every feature the real one has (GQA, Llama 3 RoPE scaling, tied
# embeddings off) so model tests run on CPU in milliseconds.
TINY_RAW = {
    "architectures": ["LlamaForCausalLM"],
    "vocab_size": 128,
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "max_position_embeddings": 256,
    "rms_norm_eps": 1e-5,
    "rope_theta": 500000.0,
    "rope_scaling": {
        "rope_type": "llama3",
        "factor": 8.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_max_position_embeddings": 64,
    },
    "tie_word_embeddings": False,
    "eos_token_id": 1,
    "bos_token_id": 0,
    "torch_dtype": "float32",
}


@pytest.fixture
def tiny_cfg() -> ModelConfig:
    return ModelConfig.from_dict(TINY_RAW)

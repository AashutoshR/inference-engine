import json

import pytest

from engine.config import (
    ModelConfig,
    estimate_param_count,
    kv_bytes_per_token,
    load_stop_token_ids,
    max_kv_tokens,
)
from tests.conftest import CONFIG_1B, CONFIG_8B, TINY_RAW


def test_parses_llama_3_2_1b():
    cfg = ModelConfig.from_pretrained(CONFIG_1B)
    assert cfg.head_dim == 64
    assert cfg.num_kv_groups == 4
    assert cfg.tie_word_embeddings
    assert cfg.rope_scaling is not None and cfg.rope_scaling.factor == 32.0
    assert cfg.eos_token_ids == (128001, 128008, 128009)


def test_head_dim_derived_when_missing():
    cfg = ModelConfig.from_pretrained(CONFIG_8B)  # 8B config has no head_dim key
    assert cfg.head_dim == 4096 // 32 == 128


@pytest.mark.parametrize(
    ("config_dir", "expected"),
    [(CONFIG_1B, 1_235_814_400), (CONFIG_8B, 8_030_261_248)],
)
def test_param_count_matches_published_models(config_dir, expected):
    assert estimate_param_count(ModelConfig.from_pretrained(config_dir)) == expected


def test_kv_bytes_per_token():
    assert kv_bytes_per_token(ModelConfig.from_pretrained(CONFIG_1B)) == 32 * 1024
    assert kv_bytes_per_token(ModelConfig.from_pretrained(CONFIG_8B)) == 128 * 1024


def test_8b_bf16_does_not_fit_on_8_gib():
    cfg = ModelConfig.from_pretrained(CONFIG_8B)
    assert max_kv_tokens(cfg, gpu_mem_gib=8, weight_bytes_per_param=2.0) == 0
    assert max_kv_tokens(cfg, gpu_mem_gib=8, weight_bytes_per_param=0.5) > 0


def test_rejects_unsupported_architecture():
    with pytest.raises(ValueError, match="unsupported architecture"):
        ModelConfig.from_dict({**TINY_RAW, "architectures": ["MixtralForCausalLM"]})


def test_rejects_unknown_rope_scaling():
    raw = {**TINY_RAW, "rope_scaling": {"rope_type": "yarn", "factor": 4.0}}
    with pytest.raises(ValueError, match="rope_scaling"):
        ModelConfig.from_dict(raw)


def test_stop_ids_merge_generation_config(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({**TINY_RAW, "eos_token_id": 128001}))
    (tmp_path / "generation_config.json").write_text(json.dumps({"eos_token_id": [128001, 128009]}))
    assert load_stop_token_ids(tmp_path) == (128001, 128009)

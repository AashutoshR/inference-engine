"""Model configuration parsed from a Hugging Face ``config.json``, plus memory estimates.

This module has no torch dependency on purpose.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SUPPORTED_ARCHITECTURES = frozenset({"LlamaForCausalLM"})


@dataclass(frozen=True)
class RopeScaling:
    """Llama 3 frequency scaling for RoPE (``rope_type == "llama3"``).

    Low-frequency components are divided by ``factor``, high-frequency ones are left
    alone, and the band between is interpolated. See the Week 1 RoPE task.
    """

    rope_type: str
    factor: float
    low_freq_factor: float = 1.0
    high_freq_factor: float = 4.0
    original_max_position_embeddings: int = 8192

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> RopeScaling:
        rope_type = raw.get("rope_type") or raw.get("type")
        if rope_type != "llama3":
            raise ValueError(f"unsupported rope_scaling type {rope_type!r}; only 'llama3' is")
        return cls(
            rope_type=rope_type,
            factor=float(raw["factor"]),
            low_freq_factor=float(raw.get("low_freq_factor", 1.0)),
            high_freq_factor=float(raw.get("high_freq_factor", 4.0)),
            original_max_position_embeddings=int(raw.get("original_max_position_embeddings", 8192)),
        )


@dataclass(frozen=True)
class ModelConfig:
    architecture: str
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    max_position_embeddings: int
    rms_norm_eps: float
    rope_theta: float
    rope_scaling: RopeScaling | None
    tie_word_embeddings: bool
    bos_token_id: int | None
    eos_token_ids: tuple[int, ...]
    torch_dtype: str

    @property
    def num_kv_groups(self) -> int:
        """Query heads that share one KV head (grouped-query attention)."""
        return self.num_attention_heads // self.num_key_value_heads

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ModelConfig:
        archs = raw.get("architectures") or []
        arch = archs[0] if archs else str(raw.get("model_type", "unknown"))
        if arch not in SUPPORTED_ARCHITECTURES:
            raise ValueError(
                f"unsupported architecture {arch!r}; supported: {sorted(SUPPORTED_ARCHITECTURES)}"
            )

        hidden = int(raw["hidden_size"])
        n_heads = int(raw["num_attention_heads"])
        n_kv = int(raw.get("num_key_value_heads") or n_heads)
        if n_heads % n_kv:
            raise ValueError(f"num_attention_heads ({n_heads}) not divisible by KV heads ({n_kv})")

        scaling = raw.get("rope_scaling")
        return cls(
            architecture=arch,
            vocab_size=int(raw["vocab_size"]),
            hidden_size=hidden,
            intermediate_size=int(raw["intermediate_size"]),
            num_hidden_layers=int(raw["num_hidden_layers"]),
            num_attention_heads=n_heads,
            num_key_value_heads=n_kv,
            head_dim=int(raw.get("head_dim") or hidden // n_heads),
            max_position_embeddings=int(raw.get("max_position_embeddings", 2048)),
            rms_norm_eps=float(raw.get("rms_norm_eps", 1e-5)),
            rope_theta=float(raw.get("rope_theta", 10000.0)),
            rope_scaling=RopeScaling.from_dict(scaling) if scaling else None,
            tie_word_embeddings=bool(raw.get("tie_word_embeddings", False)),
            bos_token_id=raw.get("bos_token_id"),
            eos_token_ids=_as_id_tuple(raw.get("eos_token_id")),
            torch_dtype=str(raw.get("torch_dtype", "bfloat16")),
        )

    @classmethod
    def from_pretrained(cls, model_dir: str | Path) -> ModelConfig:
        with open(Path(model_dir) / "config.json", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


def _as_id_tuple(value: int | list[int] | None) -> tuple[int, ...]:
    if value is None:
        return ()
    if isinstance(value, list):
        return tuple(int(v) for v in value)
    return (int(value),)


def load_stop_token_ids(model_dir: str | Path) -> tuple[int, ...]:
    """All stop IDs: ``config.json``'s EOS plus ``generation_config.json``'s.

    Llama 3 Instruct ends a turn with ``<|eot_id|>``, which may only be listed in
    ``generation_config.json``. Checking EOS alone makes generation never stop.
    """
    model_dir = Path(model_dir)
    ids = list(ModelConfig.from_pretrained(model_dir).eos_token_ids)
    gen_path = model_dir / "generation_config.json"
    if gen_path.exists():
        with open(gen_path, encoding="utf-8") as f:
            ids.extend(_as_id_tuple(json.load(f).get("eos_token_id")))
    return tuple(dict.fromkeys(ids))  # de-duplicate, keep order


# --- Memory estimates ---------------------------------------------------------------------


def estimate_param_count(cfg: ModelConfig) -> int:
    """Exact parameter count for a Llama-architecture model (no biases)."""
    h, kv_dim = cfg.hidden_size, cfg.num_key_value_heads * cfg.head_dim
    q_dim = cfg.num_attention_heads * cfg.head_dim
    attn = h * q_dim + 2 * h * kv_dim + q_dim * h
    mlp = 3 * h * cfg.intermediate_size
    norms = 2 * h
    per_layer = attn + mlp + norms
    embed = cfg.vocab_size * h
    lm_head = 0 if cfg.tie_word_embeddings else cfg.vocab_size * h
    return embed + cfg.num_hidden_layers * per_layer + h + lm_head


def kv_bytes_per_token(cfg: ModelConfig, dtype_bytes: int = 2) -> int:
    """KV cache bytes for one token across all layers (K and V)."""
    return 2 * cfg.num_hidden_layers * cfg.num_key_value_heads * cfg.head_dim * dtype_bytes


def max_kv_tokens(
    cfg: ModelConfig,
    gpu_mem_gib: float,
    weight_bytes_per_param: float = 2.0,
    reserve_gib: float = 1.0,
) -> int:
    """Tokens of KV cache that fit after weights and a reserve for activations/CUDA context."""
    weights = estimate_param_count(cfg) * weight_bytes_per_param
    free = gpu_mem_gib * 2**30 - weights - reserve_gib * 2**30
    return max(0, int(free // kv_bytes_per_token(cfg)))

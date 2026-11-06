"""Load Hugging Face safetensors weights into ``LlamaForCausalLM`` (Week 1)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

    from engine.model.llama import LlamaForCausalLM

log = logging.getLogger(__name__)


def find_safetensors_files(model_dir: str | Path) -> list[Path]:
    """Weight shards in load order: the sharded index if present, else ``model.safetensors``."""
    model_dir = Path(model_dir)
    index = model_dir / "model.safetensors.index.json"
    if index.exists():
        with open(index, encoding="utf-8") as f:
            shard_names = sorted(set(json.load(f)["weight_map"].values()))
        return [model_dir / name for name in shard_names]
    single = model_dir / "model.safetensors"
    if single.exists():
        return [single]
    raise FileNotFoundError(f"no safetensors weights in {model_dir}")


def load_model(
    model_dir: str | Path, device: str | torch.device = "cuda", dtype: torch.dtype | None = None
) -> LlamaForCausalLM:
    """Build the model and load its weights.

    Steps (Week 1):
      1. cfg = ModelConfig.from_pretrained(model_dir).
      2. Build LlamaForCausalLM(cfg) under ``torch.device("meta")`` so no memory is
         allocated for random init (an 8B model would otherwise fill RAM twice).
      3. For each shard from find_safetensors_files: safetensors.torch.load_file(shard,
         device=str(device)), cast to dtype (default bfloat16), collect into one dict.
      4. If cfg.tie_word_embeddings: state["lm_head.weight"] = state["model.embed_tokens.weight"].
      5. model.load_state_dict(state, strict=True, assign=True); then rebuild the rotary
         buffer on the real device (it was created on meta).
      6. model.eval(); log parameter count and torch.cuda.memory_allocated().
    """
    import torch
    from safetensors.torch import load_file

    from engine.config import ModelConfig
    from engine.model.layers import RotaryEmbedding
    from engine.model.llama import LlamaForCausalLM

    dtype = dtype or torch.bfloat16
    device = torch.device(device)
    cfg = ModelConfig.from_pretrained(model_dir)

    with torch.device("meta"):
        model = LlamaForCausalLM(cfg)

    state: dict[str, torch.Tensor] = {}
    for shard in find_safetensors_files(model_dir):
        for name, tensor in load_file(shard, device=str(device)).items():
            state[name] = tensor.to(dtype)
    if cfg.tie_word_embeddings:
        state["lm_head.weight"] = state["model.embed_tokens.weight"]

    model.load_state_dict(state, strict=True, assign=True)
    if cfg.tie_word_embeddings:  # one Parameter, not two that happen to share storage
        model.lm_head.weight = model.model.embed_tokens.weight
    # The rotary buffer is not in the checkpoint and was built on meta; rebuild it for real.
    # It stays float32 whatever the weight dtype, because RoPE angles need the precision.
    model.rotary = RotaryEmbedding(cfg).to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    msg = f"loaded {n_params / 1e9:.2f} B parameters as {dtype} on {device}"
    if device.type == "cuda":
        msg += f", {torch.cuda.memory_allocated(device) / 2**30:.2f} GiB allocated"
    log.info(msg)
    return model

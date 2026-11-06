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
    model_dir: str | Path,
    device: str | torch.device = "cuda",
    dtype: torch.dtype | None = None,
    quantization: str | None = None,
    group_size: int = 128,
) -> LlamaForCausalLM:
    """Build the model and load its weights.

    Steps (Week 1):
      1. cfg = ModelConfig.from_pretrained(model_dir).
      2. Build LlamaForCausalLM(cfg) under ``torch.device("meta")`` so no memory is
         allocated for random init (an 8B model would otherwise fill RAM twice).
      3. For each shard from find_safetensors_files: load it on the CPU, cast each tensor
         to dtype (default bfloat16), move it to the device, collect into one dict.
      4. If cfg.tie_word_embeddings: state["lm_head.weight"] = state["model.embed_tokens.weight"].
      5. model.load_state_dict(state, strict=True, assign=True); then rebuild the rotary
         buffer on the real device (it was created on meta).
      6. model.eval(); log parameter count and torch.cuda.memory_allocated().

    quantization="int4": every decoder-layer linear is packed to INT4 (engine.quant.int4)
    as it streams in, one tensor at a time, so the BF16 model never has to fit on the GPU.
    """
    import torch
    from safetensors.torch import load_file

    from engine.config import ModelConfig
    from engine.model.layers import RotaryEmbedding
    from engine.model.llama import LlamaForCausalLM
    from engine.quant.int4 import install, is_quantizable, make_quantized

    dtype = dtype or torch.bfloat16
    device = torch.device(device)
    if quantization not in (None, "int4", "fp8"):
        raise ValueError(f"unknown quantization {quantization!r}; supported: int4, fp8")
    if quantization and (dtype != torch.bfloat16 or device.type != "cuda"):
        raise ValueError(f"{quantization} needs dtype=bfloat16 on CUDA")
    cfg = ModelConfig.from_pretrained(model_dir)

    with torch.device("meta"):
        model = LlamaForCausalLM(cfg)

    # Read each shard on the CPU and send every tensor to the device already in its final
    # dtype. Loading onto the GPU first and casting there would hold both copies at once
    # (BF16 checkpoint + FP32 weights = 7.3 GB for the 1B model), and PyTorch's caching
    # allocator would keep that peak reserved: on an 8 GB card Windows then spills GPU
    # memory into system RAM and everything slows down by an order of magnitude.
    state: dict[str, torch.Tensor] = {}
    quantized: set[str] = set()
    for shard in find_safetensors_files(model_dir):
        for name, tensor in load_file(shard, device="cpu").items():
            weight = tensor.to(dtype=dtype).to(device)
            if quantization and is_quantizable(name):
                # Quantize on the GPU; the BF16 copy is freed right after.
                install(model, name, make_quantized(quantization, weight, group_size))
                quantized.add(name)
                del weight
            else:
                state[name] = weight
    if cfg.tie_word_embeddings:
        state["lm_head.weight"] = state["model.embed_tokens.weight"]

    result = model.load_state_dict(state, strict=not quantized, assign=True)
    if quantized:
        quant_buffers = ("packed", "scales_and_zeros", "weight_fp8", "weight_scale")
        missing = [k for k in result.missing_keys if not k.endswith(quant_buffers)]
        if missing or result.unexpected_keys:
            raise RuntimeError(f"missing {missing}, unexpected {result.unexpected_keys}")
    if cfg.tie_word_embeddings:  # one Parameter, not two that happen to share storage
        model.lm_head.weight = model.model.embed_tokens.weight
    # The rotary buffer is not in the checkpoint and was built on meta; rebuild it for real.
    # It stays float32 whatever the weight dtype, because RoPE angles need the precision.
    model.rotary = RotaryEmbedding(cfg).to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    n_params += sum(
        m.in_features * m.out_features
        for m in model.modules()
        if hasattr(m, "packed") or hasattr(m, "weight_fp8")
    )
    kind = str(dtype)
    if quantized:
        kind = f"int4 (group {group_size})" if quantization == "int4" else "fp8 (e4m3)"
    msg = f"loaded {n_params / 1e9:.2f} B parameters as {kind} on {device}"
    if device.type == "cuda":
        msg += f", {torch.cuda.memory_allocated(device) / 2**30:.2f} GiB allocated"
    log.info(msg)
    return model

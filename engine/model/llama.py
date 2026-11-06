"""The Llama network. Attribute names match HF so ``load_state_dict`` works unchanged:

model.embed_tokens.weight
model.layers.{i}.self_attn.{q,k,v,o}_proj.weight
model.layers.{i}.mlp.{gate,up,down}_proj.weight
model.layers.{i}.{input,post_attention}_layernorm.weight
model.norm.weight
lm_head.weight            (absent when tie_word_embeddings is true)
"""

from __future__ import annotations

import torch
from torch import nn

from engine.config import ModelConfig
from engine.model.layers import DecoderLayer, RMSNorm, RotaryEmbedding
from engine.runtime.kv_cache import ContiguousKVCache
from engine.runtime.paged_cache import PagedBatch


class LlamaBackbone(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(DecoderLayer(cfg, i) for i in range(cfg.num_hidden_layers))
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)


class LlamaForCausalLM(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.model = LlamaBackbone(cfg)
        self.rotary = RotaryEmbedding(cfg)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

    @torch.inference_mode()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        cache: ContiguousKVCache | PagedBatch | None = None,
        start_pos: int = 0,
        attention_mask: torch.Tensor | None = None,
        last_only: bool = False,
    ) -> torch.Tensor:
        """input_ids, positions: [B, T] -> logits [B, T, vocab] in float32.

        Prefill: T = prompt length, start_pos = 0.
        Decode:  T = 1, start_pos = tokens already in the cache.
        last_only: return logits for the last position only ([B, 1, vocab]). Generation
        needs nothing else, and full-prompt logits are big: 2,000 tokens x 128K vocab in
        float32 is 1 GB.

        Paged mode (cache is a PagedBatch): input_ids and positions are [1, N], every
        sequence's new tokens concatenated. Returns each sequence's last-token logits,
        [B, 1, vocab].
        """
        cos, sin = self.rotary(positions)
        x = self.model.embed_tokens(input_ids)
        for layer in self.model.layers:
            x = layer(x, cos, sin, cache, start_pos, attention_mask)
        if isinstance(cache, PagedBatch):
            x = x[0, cache.last_index][:, None]
        elif last_only:
            x = x[:, -1:]
        return self.lm_head(self.model.norm(x)).float()

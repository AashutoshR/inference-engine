"""Chat prompt to token ids, across transformers versions."""

from __future__ import annotations

from typing import Any


def chat_prompt_ids(tokenizer: Any, messages: list[dict[str, str]]) -> list[int]:
    """Apply the model's chat template and return plain token ids, ready for a reply.

    transformers 5 returns a BatchEncoding from apply_chat_template by default, while 4.x
    returns a list, so ask for no dict and unwrap if one comes back anyway.
    """
    out = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_dict=False
    )
    if hasattr(out, "keys"):  # BatchEncoding or dict
        out = out["input_ids"]
    return [int(t) for t in out]

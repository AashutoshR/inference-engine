"""Stream text token by token without splitting UTF-8 characters (Week 2)."""

from __future__ import annotations

from typing import Any


class IncrementalDetokenizer:
    """Decode a growing list of token ids into text deltas.

    Approach: keep ``prefix_offset`` and ``read_offset`` into the id list. On each new
    token, decode ids[prefix_offset:] and ids[prefix_offset:read_offset]; emit the
    difference only if it does not end in U+FFFD (an incomplete multi-byte character).
    Decoding a window, not single tokens, keeps SentencePiece/byte-level spacing right.
    """

    def __init__(self, tokenizer: Any, prompt_ids: list[int] | None = None) -> None:
        self.tokenizer = tokenizer
        self.ids: list[int] = list(prompt_ids or [])
        self.prefix_offset = max(len(self.ids) - 5, 0)
        self.read_offset = len(self.ids)

    def add(self, token_id: int) -> str:
        """Append one token and return the newly completed text (possibly "")."""
        self.ids.append(token_id)
        prefix_text = self._decode(self.ids[self.prefix_offset : self.read_offset])
        new_text = self._decode(self.ids[self.prefix_offset :])
        if len(new_text) <= len(prefix_text) or new_text.endswith("�"):
            return ""  # nothing new yet, or a multi-byte character is still incomplete
        self.prefix_offset = self.read_offset
        self.read_offset = len(self.ids)
        return new_text[len(prefix_text) :]

    def _decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(
            ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

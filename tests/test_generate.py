"""Week 2 definition of done: detokenizer, loader and the generation loop, on CPU."""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from engine.model.llama import LlamaForCausalLM  # noqa: E402
from engine.runtime.detokenizer import IncrementalDetokenizer  # noqa: E402
from engine.runtime.generate import generate  # noqa: E402
from engine.runtime.params import SamplingParams  # noqa: E402
from tests.conftest import TINY_RAW  # noqa: E402


class ByteTokenizer:
    """One token per UTF-8 byte: the hardest case for streaming, since characters split."""

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, ids: list[int], **_: object) -> str:
        return bytes(ids).decode("utf-8", errors="replace")


@pytest.fixture
def tiny_model(tiny_cfg):
    torch.manual_seed(0)
    return LlamaForCausalLM(tiny_cfg).eval()


def tokens_of(outputs, row: int = 0) -> list[int]:
    return [o.token_id for o in outputs if o.seq_index == row]


# --- Detokenizer --------------------------------------------------------------------------


def test_detokenizer_never_splits_a_character():
    tok = ByteTokenizer()
    text = "héllo wörld 😀 done"
    detok = IncrementalDetokenizer(tok, tok.encode("prompt: "))
    deltas = [detok.add(t) for t in tok.encode(text)]
    assert "".join(deltas) == text
    assert not any("�" in d for d in deltas)


# --- Loader -------------------------------------------------------------------------------


@pytest.mark.parametrize("tied", [False, True])
def test_load_model_matches_hf(tmp_path, tied):
    transformers = pytest.importorskip("transformers")
    from engine.model.loader import load_model
    from tests.test_layers import _hf_kwargs

    raw = {**TINY_RAW, "tie_word_embeddings": tied}
    torch.manual_seed(0)
    hf_cfg = transformers.LlamaConfig(**{**_hf_kwargs(), "tie_word_embeddings": tied})
    hf = transformers.LlamaForCausalLM(hf_cfg).eval()
    hf.save_pretrained(tmp_path)
    # Use the original-format config.json, as a downloaded checkpoint has it.
    (tmp_path / "config.json").write_text(json.dumps(raw))

    ours = load_model(tmp_path, device="cpu", dtype=torch.float32)
    assert not any(p.is_meta for p in ours.parameters())
    assert ours.rotary.inv_freq.device.type == "cpu"
    if tied:
        assert ours.lm_head.weight is ours.model.embed_tokens.weight

    ids = torch.randint(0, raw["vocab_size"], (1, 9))
    with torch.no_grad():
        expected = hf(input_ids=ids).logits.float()
    got = ours(ids, torch.arange(9).unsqueeze(0))
    torch.testing.assert_close(got, expected, atol=1e-4, rtol=1e-4)


def test_load_model_casts_to_bf16(tmp_path):
    transformers = pytest.importorskip("transformers")
    from engine.model.loader import load_model
    from tests.test_layers import _hf_kwargs

    transformers.LlamaForCausalLM(transformers.LlamaConfig(**_hf_kwargs())).save_pretrained(
        tmp_path
    )
    (tmp_path / "config.json").write_text(json.dumps(TINY_RAW))
    model = load_model(tmp_path, device="cpu")
    assert all(p.dtype == torch.bfloat16 for p in model.parameters())
    assert model.rotary.inv_freq.dtype == torch.float32


# --- Generation ---------------------------------------------------------------------------


def test_greedy_matches_naive_full_forward(tiny_model):
    prompt = [3, 17, 42, 9, 100]
    params = SamplingParams(temperature=0, max_tokens=10)
    got = tokens_of(generate(tiny_model, ByteTokenizer(), [prompt], params, ()))

    ids = list(prompt)
    for _ in range(10):  # no cache: recompute the whole sequence every step
        x = torch.tensor([ids])
        ids.append(int(tiny_model(x, torch.arange(len(ids)).unsqueeze(0))[0, -1].argmax()))
    assert got == ids[len(prompt) :]


def test_batched_greedy_equals_single_prompt(tiny_model):
    prompts = [[5, 6, 7], [10, 11, 12, 13, 14, 15, 16], [99]]
    params = SamplingParams(temperature=0, max_tokens=12)
    batched = list(generate(tiny_model, ByteTokenizer(), prompts, params, ()))
    for row, prompt in enumerate(prompts):
        single = list(generate(tiny_model, ByteTokenizer(), [prompt], params, ()))
        assert tokens_of(batched, row) == tokens_of(single)


def test_max_tokens_finishes_with_length(tiny_model):
    outs = list(
        generate(tiny_model, ByteTokenizer(), [[1, 2, 3]], SamplingParams(max_tokens=7), ())
    )
    assert len(outs) == 7
    assert [o.finished for o in outs] == [False] * 6 + [True]
    assert outs[-1].finish_reason == "length"


def test_stop_token_ends_sequence_and_is_returned(tiny_model):
    params = SamplingParams(temperature=0, max_tokens=10)
    free = tokens_of(generate(tiny_model, ByteTokenizer(), [[4, 5]], params, ()))
    stop_id = free[3]
    cut = free.index(stop_id)  # first time the stop id appears

    outs = list(generate(tiny_model, ByteTokenizer(), [[4, 5]], params, (stop_id,)))
    assert tokens_of(outs) == free[: cut + 1]
    assert outs[-1].finish_reason == "stop"


def test_stop_string_is_not_emitted(tiny_model):
    tok = ByteTokenizer()
    prompt = tok.encode("Say: ")
    free = list(
        generate(tiny_model, tok, [prompt], SamplingParams(temperature=0, max_tokens=20), ())
    )
    full = "".join(o.text for o in free)
    stop = full[6:8]
    assert len(stop) == 2

    params = SamplingParams(temperature=0, max_tokens=20, stop=(stop,))
    outs = list(generate(tiny_model, tok, [prompt], params, ()))
    assert "".join(o.text for o in outs) == full[: full.find(stop)]
    assert outs[-1].finish_reason == "stop"


def test_same_seed_reproduces_sampled_output(tiny_model):
    params = SamplingParams(temperature=1.0, top_p=0.9, max_tokens=15, seed=1234)
    a = tokens_of(generate(tiny_model, ByteTokenizer(), [[8, 9]], params, ()))
    b = tokens_of(generate(tiny_model, ByteTokenizer(), [[8, 9]], params, ()))
    assert a == b


def test_repetition_penalty_changes_greedy_choice():
    from engine.runtime.sampler import sample

    logits = torch.tensor([[2.0, 1.9, 0.0]])
    prev = torch.tensor([[0]])
    params = SamplingParams(temperature=0, repetition_penalty=1.5)
    assert sample(logits, params, prev_tokens=prev).item() == 1

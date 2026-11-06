import pytest

from engine.runtime.params import SamplingParams
from tests.reference import first_divergence, is_near_tie


def test_defaults_are_valid():
    assert not SamplingParams().greedy
    assert SamplingParams(temperature=0).greedy


@pytest.mark.parametrize(
    "kwargs",
    [
        {"temperature": -0.1},
        {"top_k": -1},
        {"top_p": 0.0},
        {"top_p": 1.5},
        {"min_p": 1.0},
        {"repetition_penalty": 0},
        {"max_tokens": 0},
    ],
)
def test_invalid_params_rejected(kwargs):
    with pytest.raises(ValueError):
        SamplingParams(**kwargs)


def test_first_divergence():
    assert first_divergence([1, 2, 3], [1, 2, 3]) is None
    assert first_divergence([1, 2, 3], [1, 9, 3]) == 1
    assert first_divergence([1, 2], [1, 2, 3]) == 2


def test_near_tie_threshold():
    assert is_near_tie(0.005)
    assert not is_near_tie(0.5)


def test_chat_prompt_ids_unwraps_batch_encoding():
    from engine.runtime.chat import chat_prompt_ids

    class DictTokenizer:  # transformers 5 style: returns a mapping
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs["add_generation_prompt"] is True
            return {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}

    class ListTokenizer:  # transformers 4 style: returns ids
        def apply_chat_template(self, messages, **kwargs):
            return [4, 5]

    msgs = [{"role": "user", "content": "hi"}]
    assert chat_prompt_ids(DictTokenizer(), msgs) == [1, 2, 3]
    assert chat_prompt_ids(ListTokenizer(), msgs) == [4, 5]

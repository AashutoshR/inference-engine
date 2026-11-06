"""Gate 1: greedy output matches Hugging Face token for token (Week 3).

Needs real weights:  ENGINE_MODEL_DIR=models/Llama-3.2-1B-Instruct pytest -m slow
Grow tests/data/prompts.jsonl to 200 prompts before calling the gate passed.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tests.reference import first_divergence

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

MODEL_DIR = os.environ.get("ENGINE_MODEL_DIR")
PROMPTS = [
    json.loads(line)
    for line in (Path(__file__).parent / "data" / "prompts.jsonl").read_text("utf-8").splitlines()
    if line.strip()
]
NEW_TOKENS = 128

pytestmark = [
    pytest.mark.slow,
    pytest.mark.gpu,
    pytest.mark.skipif(not MODEL_DIR, reason="set ENGINE_MODEL_DIR to a downloaded model"),
]


@pytest.fixture(scope="module")
def setup():
    from engine.config import load_stop_token_ids
    from engine.model.loader import load_model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32  # exact-match mode; BF16 runs use the near-tie rule instead
    tok = transformers.AutoTokenizer.from_pretrained(MODEL_DIR)
    hf = transformers.AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=dtype).to(device)
    ours = load_model(MODEL_DIR, device=device, dtype=dtype)
    return tok, hf.eval(), ours, load_stop_token_ids(MODEL_DIR), device


@pytest.mark.parametrize("prompt", PROMPTS, ids=[p["id"] for p in PROMPTS])
def test_greedy_matches_hf(setup, prompt):
    from engine.runtime.chat import chat_prompt_ids
    from engine.runtime.generate import generate
    from engine.runtime.params import SamplingParams

    tok, hf, ours, stop_ids, device = setup
    ids = chat_prompt_ids(tok, prompt["messages"])

    with torch.no_grad():
        ref = hf.generate(
            torch.tensor([ids], device=device),
            max_new_tokens=NEW_TOKENS,
            do_sample=False,
            eos_token_id=list(stop_ids),
        )[0, len(ids) :].tolist()

    params = SamplingParams(temperature=0, max_tokens=NEW_TOKENS)
    got = [out.token_id for out in generate(ours, tok, [ids], params, stop_ids)]

    pos = first_divergence(got, ref)
    assert pos is None, (
        f"diverged at token {pos}: ours={got[pos : pos + 5]} hf={ref[pos : pos + 5]} "
        f"context={tok.decode(ref[max(0, pos - 10) : pos])!r}"
    )

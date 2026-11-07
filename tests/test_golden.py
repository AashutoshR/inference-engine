"""Gate 1: greedy output matches Hugging Face token for token (Week 3).

    # FP32, exact match required (Gate 1, check 1)
    ENGINE_MODEL_DIR=models/Llama-3.2-1B-Instruct pytest -m slow
    # BF16, every mismatch must be a near-tie (Gate 1, check 2)
    ENGINE_MODEL_DIR=models/Llama-3.2-3B-Instruct ENGINE_GOLDEN_DTYPE=bfloat16 pytest -m slow

ENGINE_GOLDEN_LIMIT=N runs only the first N prompts.

The Hugging Face reference (tokens, plus the top-2 logit gap at every step) is computed
once and cached in .cache/golden/, keyed by model, dtype, device, prompts and library
versions. Only one model is on the GPU at a time, so FP32 1B (5 GB) fits on an 8 GB card.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import warnings
from pathlib import Path

import pytest

from tests.reference import first_divergence, is_near_tie

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = os.environ.get("ENGINE_MODEL_DIR")
DTYPE_NAME = os.environ.get("ENGINE_GOLDEN_DTYPE", "float32")
EXACT = DTYPE_NAME == "float32"
LIMIT = int(os.environ.get("ENGINE_GOLDEN_LIMIT", "0")) or None
NEW_TOKENS = 128
PROMPTS = [
    json.loads(line)
    for line in (Path(__file__).parent / "data" / "prompts.jsonl").read_text("utf-8").splitlines()
    if line.strip()
][:LIMIT]

pytestmark = [
    pytest.mark.slow,
    pytest.mark.gpu,
    pytest.mark.skipif(not MODEL_DIR, reason="set ENGINE_MODEL_DIR to a downloaded model"),
]


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _cache_path() -> Path:
    key = json.dumps(
        [
            Path(MODEL_DIR).resolve().name,
            DTYPE_NAME,
            _device(),
            NEW_TOKENS,
            transformers.__version__,
            torch.__version__,
            PROMPTS,
        ],
        ensure_ascii=False,
    )
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    name = f"{Path(MODEL_DIR).resolve().name}-{DTYPE_NAME}-{_device()}-{digest}.json"
    return ROOT / ".cache" / "golden" / name


def _free_gpu() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


OUTCOMES: list[dict] = []  # one row per prompt, written to bench/results at the end


@pytest.fixture(scope="module", autouse=True)
def report():
    """Write every prompt's outcome to bench/results/gate1-<model>-<dtype>.json."""
    import time

    start = time.time()
    yield
    if not OUTCOMES:
        return
    out = ROOT / "bench" / "results" / f"gate1-{Path(MODEL_DIR).resolve().name}-{DTYPE_NAME}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    exact = sum(o["result"] == "exact" for o in OUTCOMES)
    ties = sum(o["result"] == "near-tie" for o in OUTCOMES)
    summary = {
        "model": Path(MODEL_DIR).resolve().name,
        "dtype": DTYPE_NAME,
        "device": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
        "rule": "every token identical" if EXACT else "mismatches must be near-ties (gap < 1e-2)",
        "prompts": len(OUTCOMES),
        "exact": exact,
        "near_ties": ties,
        "failed": len(OUTCOMES) - exact - ties,
        "tokens_compared": sum(o["tokens"] for o in OUTCOMES),
        "seconds": round(time.time() - start, 1),
        "outcomes": OUTCOMES,
    }
    out.write_text(json.dumps(summary, indent=1) + chr(10), encoding="utf-8")


@pytest.fixture(scope="module")
def tokenizer():
    return transformers.AutoTokenizer.from_pretrained(MODEL_DIR)


@pytest.fixture(scope="module")
def stop_ids():
    from engine.config import load_stop_token_ids

    return load_stop_token_ids(MODEL_DIR)


@pytest.fixture(scope="module")
def reference(tokenizer, stop_ids):
    """{prompt id: {"tokens": [...], "gaps": [...]}} from HF generate, cached on disk."""
    from engine.runtime.chat import chat_prompt_ids

    path = _cache_path()
    if path.exists():
        return json.loads(path.read_text("utf-8"))

    device, dtype = _device(), getattr(torch, DTYPE_NAME)
    hf = transformers.AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=dtype).to(device)
    hf.eval()
    refs = {}
    with torch.inference_mode():
        for prompt in PROMPTS:
            ids = torch.tensor([chat_prompt_ids(tokenizer, prompt["messages"])], device=device)
            out = hf.generate(
                ids,
                attention_mask=torch.ones_like(ids),
                max_new_tokens=NEW_TOKENS,
                do_sample=False,
                temperature=None,
                top_p=None,
                eos_token_id=list(stop_ids),
                pad_token_id=stop_ids[0],
                output_logits=True,
                return_dict_in_generate=True,
            )
            top2 = [step[0].float().topk(2).values for step in out.logits]
            refs[prompt["id"]] = {
                "tokens": out.sequences[0, ids.shape[1] :].tolist(),
                "gaps": [float(t[0] - t[1]) for t in top2],
            }
    del hf, out  # `out` holds HF's KV cache for the last prompt
    _free_gpu()

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(refs), "utf-8")
    return refs


@pytest.fixture(scope="module")
def ours(reference):  # depends on `reference` so HF is off the GPU before this loads
    from engine.model.loader import load_model

    model = load_model(MODEL_DIR, device=_device(), dtype=getattr(torch, DTYPE_NAME))
    yield model
    del model
    _free_gpu()


@pytest.mark.parametrize("prompt", PROMPTS, ids=[p["id"] for p in PROMPTS])
def test_greedy_matches_hf(reference, ours, tokenizer, stop_ids, prompt):
    from engine.runtime.chat import chat_prompt_ids
    from engine.runtime.generate import generate
    from engine.runtime.params import SamplingParams

    ref = reference[prompt["id"]]
    ids = chat_prompt_ids(tokenizer, prompt["messages"])
    params = SamplingParams(temperature=0, max_tokens=NEW_TOKENS)
    got = [out.token_id for out in generate(ours, tokenizer, [ids], params, stop_ids)]

    pos = first_divergence(got, ref["tokens"])
    row = {"id": prompt["id"], "prompt_tokens": len(ids), "tokens": len(ref["tokens"])}
    if pos is None:
        OUTCOMES.append({**row, "result": "exact"})
        return
    gap = ref["gaps"][pos] if pos < len(ref["gaps"]) else float("nan")
    ok = not EXACT and is_near_tie(gap)
    OUTCOMES.append({**row, "result": "near-tie" if ok else "failed", "diverged_at": pos,
                     "gap": gap})  # fmt: skip
    context = tokenizer.decode(ref["tokens"][max(0, pos - 10) : pos])
    msg = (
        f"{prompt['id']}: diverged at token {pos} of {len(ref['tokens'])}, "
        f"HF top-2 logit gap {gap:.2e}; ours={got[pos : pos + 5]} "
        f"hf={ref['tokens'][pos : pos + 5]} context={context!r}"
    )
    if EXACT:
        pytest.fail(msg)
    assert is_near_tie(gap), msg
    warnings.warn(f"near-tie accepted: {msg}", stacklevel=1)

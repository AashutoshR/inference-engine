"""Turn bench/results/*.json into a results page with charts.

    python bench/report.py

Writes:
  RESULTS.md                 tables and charts, rendered by GitHub
  docs/results/*.svg         the charts (fixed colors that read on light and dark GitHub)
  docs/results.html          the same report as a standalone page (follows the viewer's theme)

Missing results are shown as "not run yet", so the report can be built at any time.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "bench" / "results"
DOCS = ROOT / "docs"
CHARTS = DOCS / "results"

M1, M3 = "Llama-3.2-1B-Instruct", "Llama-3.2-3B-Instruct"
SHORT = {M1: "Llama 3.2 1B", M3: "Llama 3.2 3B"}

# Fixed colors for SVGs embedded in Markdown (GitHub shows them as images, so CSS
# variables cannot reach them): mid tones that read on both GitHub themes.
FIXED = {
    "text": "#7d8590", "strong": "#7d8590", "grid": "#7d859040", "accent": "#3b82f6",
    "accent2": "#e3a008", "dim": "#7d859066", "good": "#2ea043", "warn": "#d29922",
    "bad": "#e5534b", "font": "system-ui, -apple-system, Segoe UI, sans-serif",
}  # fmt: skip
# Theme tokens for the HTML page.
TOKENS = {
    "text": "var(--muted)", "strong": "var(--fg)", "grid": "var(--grid)", "accent": "var(--accent)",
    "accent2": "var(--accent2)", "dim": "var(--dim)", "good": "var(--good)", "warn": "var(--warn)",
    "bad": "var(--bad)", "font": "var(--font-data)",
}  # fmt: skip


# --- loading ------------------------------------------------------------------------------


def load(name: str):
    path = RES / name
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def junit(name: str) -> dict | None:
    path = RES / name
    if not path.exists():
        return None
    root = ET.parse(path).getroot()
    suite = root if root.tag == "testsuite" else root.find("testsuite")
    a = suite.attrib
    total, fail = int(a["tests"]), int(a.get("failures", 0)) + int(a.get("errors", 0))
    skipped = int(a.get("skipped", 0))
    return {"tests": total, "passed": total - fail - skipped, "failed": fail,
            "skipped": skipped, "seconds": round(float(a.get("time", 0)), 1)}  # fmt: skip


def variant_label(row: dict) -> str:
    base = {
        "static": "Phase 1 loop (no paging)",
        "eager": "Paged engine, PyTorch attention",
        "triton": "+ Triton attention kernel",
        "graphs": "+ CUDA graphs",
    }.get(row["mode"])
    if row["mode"] == "spec":
        base = (
            "Speculative, n-gram lookup" if row.get("spec") == "ngram" else "Speculative, 1B draft"
        )
    quant = {"int4": " · INT4", "fp8": " · FP8"}.get(row["quant"], "")
    return base + quant


def decode_rows(model: str) -> list[dict]:
    rows = load(f"decode-{model}.json") or []
    order = {"static": 0, "eager": 1, "triton": 2, "graphs": 3, "spec": 4}
    qorder = {"none": 0, "fp8": 1, "int4": 2}
    return sorted(rows, key=lambda r: (order[r["mode"]], qorder[r["quant"]], str(r.get("spec"))))


def load_sweep(model: str) -> list[dict]:
    rows = [load(p.name) for p in RES.glob(f"load-{model}-c*.json")]
    return sorted((r for r in rows if r), key=lambda r: r["concurrency"])


# --- SVG charts ---------------------------------------------------------------------------


def esc(s) -> str:
    return html.escape(str(s), quote=True)


def svg(width: int, height: int, body: str, title: str, pal: dict) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" role="img" aria-label="{esc(title)}" '
        f'font-family="{pal["font"]}" font-size="13">{body}</svg>'
    )


def hbars(rows: list[tuple[str, float, str]], title: str, unit: str, pal: dict,
          highlight: int | None = None, width: int = 760) -> str:  # fmt: skip
    """Horizontal bars: (label, value, note) per row; the highlighted row uses the accent."""
    label_w, top, row_h = 250, 44, 30
    bar_w = width - label_w - 150
    vmax = max((v for _, v, _ in rows), default=1) or 1
    h = top + row_h * len(rows) + 16
    out = [f'<text x="0" y="20" font-size="15" font-weight="600" fill="{pal["strong"]}">'
           f'{esc(title)}</text>']  # fmt: skip
    for i, (label, value, note) in enumerate(rows):
        y = top + i * row_h
        w = max(2, bar_w * value / vmax)
        fill = pal["accent"] if i == highlight else pal["dim"]
        out.append(f'<text x="{label_w - 12}" y="{y + 15}" text-anchor="end" '
                   f'fill="{pal["strong"]}">{esc(label)}</text>')  # fmt: skip
        out.append(f'<rect x="{label_w}" y="{y + 3}" width="{w:.1f}" height="18" rx="3" '
                   f'fill="{fill}"/>')  # fmt: skip
        out.append(f'<text x="{label_w + w + 8:.1f}" y="{y + 16}" fill="{pal["strong"]}" '
                   f'font-weight="600">{value:,.1f} {esc(unit)}</text>')  # fmt: skip
        if note:
            out.append(f'<text x="{label_w + w + 8:.1f}" y="{y + 29}" font-size="11" '
                       f'fill="{pal["text"]}">{esc(note)}</text>')  # fmt: skip
    return svg(width, h + 6, "".join(out), title, pal)


def line_chart(xs: list[int], series: list[tuple[str, list[float], str]], title: str,
               y_label: str, pal: dict, width: int = 760, height: int = 300) -> str:  # fmt: skip
    """Categorical x (evenly spaced), one or more series sharing a y scale from 0."""
    left, right, top, bottom = 64, 24, 48, 44
    pw, ph = width - left - right, height - top - bottom
    vmax = max((v for _, vals, _ in series for v in vals), default=1) or 1
    step = 10 ** math.floor(math.log10(vmax))
    for mult in (1, 2, 5, 10):
        if vmax / (step * mult) <= 5:
            step *= mult
            break
    ymax = step * math.ceil(vmax / step)
    X = lambda i: left + (pw * (i + 0.5) / len(xs))  # noqa: E731
    Y = lambda v: top + ph - ph * v / ymax  # noqa: E731
    out = [f'<text x="0" y="20" font-size="15" font-weight="600" fill="{pal["strong"]}">'
           f'{esc(title)}</text>']  # fmt: skip
    t = 0.0
    while t <= ymax + 1e-9:
        out.append(f'<line x1="{left}" x2="{width - right}" y1="{Y(t):.1f}" y2="{Y(t):.1f}" '
                   f'stroke="{pal["grid"]}"/>')  # fmt: skip
        out.append(f'<text x="{left - 8}" y="{Y(t) + 4:.1f}" text-anchor="end" font-size="11" '
                   f'fill="{pal["text"]}">{t:,.{0 if step >= 1 else 2}f}</text>')  # fmt: skip
        t += step
    for i, x in enumerate(xs):
        out.append(f'<text x="{X(i):.1f}" y="{height - bottom + 18}" text-anchor="middle" '
                   f'font-size="11" fill="{pal["text"]}">{x}</text>')  # fmt: skip
    out.append(f'<text x="{left + pw / 2}" y="{height - 6}" text-anchor="middle" font-size="11" '
               f'fill="{pal["text"]}">concurrent users</text>')  # fmt: skip
    out.append(f'<text x="12" y="{top - 14}" font-size="11" fill="{pal["text"]}">'
               f'{esc(y_label)}</text>')  # fmt: skip
    for s_i, (name, vals, color) in enumerate(series):
        pts = " ".join(f"{X(i):.1f},{Y(v):.1f}" for i, v in enumerate(vals))
        out.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        for i, v in enumerate(vals):
            out.append(f'<circle cx="{X(i):.1f}" cy="{Y(v):.1f}" r="4" fill="{color}"/>')
            out.append(f'<text x="{X(i):.1f}" y="{Y(v) - 10:.1f}" text-anchor="middle" '
                       f'font-size="11" fill="{pal["strong"]}">{v:,.{0 if v >= 10 else 2}f}</text>')  # fmt: skip
        lx = width - right - 150 * (len(series) - s_i)
        out.append(f'<rect x="{lx}" y="{top - 28}" width="12" height="12" rx="2" fill="{color}"/>'
                   f'<text x="{lx + 18}" y="{top - 18}" font-size="12" fill="{pal["strong"]}">'
                   f'{esc(name)}</text>')  # fmt: skip
    return svg(width, height, "".join(out), title, pal)


def gate_strip(g: dict, pal: dict, width: int = 760) -> str:
    """One square per prompt: green exact, amber near-tie, red failed."""
    cols, cell, top = 40, 17, 40
    rows = math.ceil(len(g["outcomes"]) / cols)
    title = (f'{SHORT.get(g["model"], g["model"])}, {g["dtype"]}: '
             f'{g["exact"]} exact, {g["near_ties"]} near-ties, {g["failed"]} failed')  # fmt: skip
    out = [f'<text x="0" y="20" font-size="15" font-weight="600" fill="{pal["strong"]}">'
           f'{esc(title)}</text>']  # fmt: skip
    colors = {"exact": pal["good"], "near-tie": pal["warn"], "failed": pal["bad"]}
    for i, o in enumerate(g["outcomes"]):
        x, y = (i % cols) * cell, top + (i // cols) * cell
        tip = o["id"] + ("" if o["result"] == "exact" else f" (token {o.get('diverged_at')})")
        out.append(f'<rect x="{x}" y="{y}" width="{cell - 3}" height="{cell - 3}" rx="2" '
                   f'fill="{colors[o["result"]]}"><title>{esc(tip)}</title></rect>')  # fmt: skip
    return svg(width, top + rows * cell + 8, "".join(out), title, pal)


# --- assembling ---------------------------------------------------------------------------


def collect() -> dict:
    gates1 = {k: load(f"gate1-{m}-{d}.json") for k, m, d in
              (("1b", M1, "float32"), ("3b", M3, "bfloat16"))}  # fmt: skip
    return {
        "env": load("environment.json") or {},
        "decode": {m: decode_rows(m) for m in (M1, M3)},
        "prefill": {m: load(f"phase1-{m}.json") for m in (M1, M3)},
        "ppl": {
            m: {q: load(f"ppl-{m}-{q}.json") for q in ("none", "int4-g128", "fp8")}
            for m in (M1, M3)
        },  # fmt: skip
        "load": load_sweep(M1),
        "soak": load("gate2-soak.json"),
        "gate1": gates1,
        "tests": {"cpu": junit("tests-cpu.xml"), "gpu": junit("tests-gpu.xml")},
        "invariance": [
            r for r in (load(f"invariance-{M1}-{t}.json") for t in ("float32", "bfloat16")) if r
        ],  # fmt: skip
    }


def charts(d: dict, pal: dict) -> dict[str, str]:
    out = {}
    for m in (M1, M3):
        rows = d["decode"][m]
        if rows:
            bars = []
            for r in rows:
                note = f"{r['itl_ms']} ms per token · {r['peak_memory_gib']} GiB"
                if "acceptance_rate" in r:
                    note += f" · {r['tokens_per_target_pass']} tokens per pass"
                bars.append((variant_label(r), r["decode_tokens_per_s"], note))
            best = max(range(len(bars)), key=lambda i: bars[i][1])
            out[f"decode-{m}"] = hbars(
                bars, f"{SHORT[m]}: one user, tokens per second", "tok/s", pal, highlight=best
            )
    if d["load"]:
        xs = [r["concurrency"] for r in d["load"]]
        out["load-throughput"] = line_chart(
            xs, [("output tokens/s", [r["output_tokens_per_s"] for r in d["load"]],
                  pal["accent"])],
            f"{SHORT[M1]}: total throughput as users are added", "tokens per second", pal,
        )  # fmt: skip
        out["load-latency"] = line_chart(
            xs,
            [("time to first token, p50", [r["ttft_s"]["p50"] for r in d["load"]], pal["accent"]),
             ("p99", [r["ttft_s"]["p99"] for r in d["load"]], pal["accent2"])],
            f"{SHORT[M1]}: how long users wait for the first word", "seconds", pal,
        )  # fmt: skip
    ppl_rows = []
    for m in (M1, M3):
        for q, label in (("none", "BF16"), ("fp8", "FP8"), ("int4-g128", "INT4")):
            r = d["ppl"][m][q]
            if r:
                base = d["ppl"][m]["none"]
                note = "" if q == "none" or not base else (
                    f'{r["perplexity"] - base["perplexity"]:+.3f} vs BF16 · '
                    f'{r["peak_memory_gib"]} GiB')  # fmt: skip
                if q == "none":
                    note = f"{r['peak_memory_gib']} GiB"
                ppl_rows.append((f"{SHORT[m]} · {label}", r["perplexity"], note))
    if ppl_rows:
        out["perplexity"] = hbars(ppl_rows, "WikiText-2 perplexity (lower is better)", "", pal)
    for k, g in d["gate1"].items():
        if g:
            out[f"gate1-{k}"] = gate_strip(g, pal)
    return out


def headline(d: dict) -> list[tuple[str, str, str]]:
    tiles = []
    best1 = max((r["decode_tokens_per_s"] for r in d["decode"][M1]), default=None)
    static1 = next((r for r in d["decode"][M1] if r["mode"] == "static"), None)
    if best1:
        sub = f"{best1 / static1['decode_tokens_per_s']:.1f}x the Phase 1 loop" if static1 else ""
        tiles.append(("One user, 1B", f"{best1:,.0f} tok/s", sub))
    if d["load"]:
        top = max(d["load"], key=lambda r: r["output_tokens_per_s"])
        tiles.append(("Many users, 1B", f'{top["output_tokens_per_s"]:,.0f} tok/s',
                      f'{top["concurrency"]} concurrent streams'))  # fmt: skip
    g = [x for x in d["gate1"].values() if x]
    if g:
        tok = sum(x["tokens_compared"] for x in g)
        ok = sum(x["exact"] + x["near_ties"] for x in g)
        tiles.append(("Matches Hugging Face", f"{ok} / {sum(x['prompts'] for x in g)}",
                      f"prompts, {tok:,} tokens compared"))  # fmt: skip
    p = d["ppl"][M3]
    if p["none"] and p["int4-g128"]:
        tiles.append(("INT4 quality cost, 3B",
                      f'{p["int4-g128"]["perplexity"] - p["none"]["perplexity"]:+.2f}',
                      f'perplexity, {p["none"]["peak_memory_gib"]} to '
                      f'{p["int4-g128"]["peak_memory_gib"]} GiB'))  # fmt: skip
    return tiles


def gate_status(d: dict) -> list[tuple[str, str, str]]:
    """(name, state, detail) with state in pass / fail / pending."""
    out = []
    for k, label in (("1b", "Gate 1 · 1B FP32 exact"), ("3b", "Gate 1 · 3B BF16 near-tie")):
        g = d["gate1"][k]
        if g is None:
            out.append((label, "pending", "not run yet"))
        else:
            state = "pass" if g["failed"] == 0 and g["prompts"] == 200 else "fail"
            out.append((label, state, f'{g["exact"]} exact, {g["near_ties"]} near-ties, '
                                      f'{g["failed"]} failed of {g["prompts"]}'))  # fmt: skip
    s = d["soak"]
    if s is None:
        out.append(("Gate 2 · 64 streams, 30 min", "pending", "not run yet"))
    else:
        ok = not s["errors"] and not s["mismatched_prompts"] and s["final_kv_used"] == 0
        out.append(("Gate 2 · 64 streams soak", "pass" if ok else "fail",
                    f'{s["duration_s"] / 60:.0f} min, {s["requests_ok"]:,} requests, '
                    f'{sum(s["errors"].values())} errors, KV left in use '
                    f'{s["final_kv_used"]:.0%}, repeat answers differed: {mismatch_note(s)}'))  # fmt: skip
    for k, label in (("cpu", "Unit tests (CPU)"), ("gpu", "GPU tests")):
        t = d["tests"][k]
        if t is None:
            out.append((label, "pending", "not run yet"))
        else:
            out.append((label, "pass" if t["failed"] == 0 else "fail",
                        f'{t["passed"]} passed, {t["failed"]} failed, {t["seconds"]} s'))  # fmt: skip
    return out


def mismatch_note(r: dict) -> str:
    """How many prompts got a different greedy answer on repeat, and how close the call was."""
    n = len(r["mismatched_prompts"])
    if not n:
        return "none"
    word = "prompt" if n == 1 else "prompts"
    splits = list(r.get("mismatch_splits", {}).values())
    gaps = sorted(v["top2_gap"] for v in splits if v["top2_gap"] is not None)
    if not gaps:
        return f"{n} {word}"
    small = sum(g <= 0.25 for g in gaps)
    return (
        f"{n} {word}; of {len(gaps)} measured, {small} split where the top two choices "
        f"were within 0.25 (largest gap {gaps[-1]:g})"
    )


def gate1_failures(d: dict) -> list[tuple[str, list[str], list[list[str]], str]]:
    """(title, headers, rows, note) for every Gate 1 run with failed prompts."""
    out = []
    for g in d["gate1"].values():
        bad = [o for o in (g or {}).get("outcomes", []) if o["result"] == "failed"]
        if not bad:
            continue
        rows = [[o["id"], str(o["prompt_tokens"]), str(o["diverged_at"]), f'{o["gap"]:g}']
                for o in bad]  # fmt: skip
        note = ""
        if g["dtype"] == "bfloat16" and {o["gap"] for o in bad} == {0.125}:
            note = ("Every gap is exactly 0.125. That is one BF16 rounding step for logits "
                    "between 16 and 32: the closest two different values BF16 can hold "
                    "there. So these are ties at BF16 precision that the 0.01 rule cannot "
                    "see, because BF16 cannot express a gap that small. They all come from "
                    "long prompts (850 to 1,950 tokens), where rounding has the most steps "
                    "to accumulate. The gate is reported as failed, as written; the fix is to "
                    "compare FP32 logits or to measure the gap in BF16 steps.")  # fmt: skip
        out.append((f'{g["model"]} {g["dtype"]}: the {len(bad)} failed prompts',
                    ["prompt", "prompt tokens", "diverged at token", "top-2 logit gap"],
                    rows, note))  # fmt: skip
    return out


PPL_HEAD = ["model", "weights", "perplexity", "vs BF16", "peak GPU memory"]


def ppl_rows(d: dict) -> list[list[str]]:
    qnames = (("none", "BF16"), ("fp8", "FP8"), ("int4-g128", "INT4, groups of 128"))
    rows = []
    for m in (M1, M3):
        base = d["ppl"][m]["none"]
        for q, label in qnames:
            r = d["ppl"][m][q]
            if not r:
                continue
            delta = f"{r['perplexity'] - base['perplexity']:+.2f}" if base and q != "none" else ""
            rows.append([SHORT[m], label, f'{r["perplexity"]:.2f}', delta,
                         f'{r["peak_memory_gib"]} GiB'])  # fmt: skip
    return rows


INV_INTRO = ("Under load, the same greedy prompt can come back with a different answer. Is "
             "that a bug or rounding? bench/batch_invariance.py runs 64 prompts through "
             "the engine on the GPU two ways: alone, one at a time with the prefix cache off, "
             "and loaded, all of them twice over with one new arrival per step, mixed "
             "prompt-and-decode steps and the prefix cache on. If batching or caching "
             "changed the math, FP32 would show it; if it only changes rounding, FP32 "
             "(65,000 times finer than BF16) should match almost everywhere.")  # fmt: skip
INV_HEAD = ["weights", "loaded requests", "largest batch", "identical", "differed",
            "largest top-2 gap at a split"]  # fmt: skip


def inv_rows(d: dict) -> list[list[str]]:
    return [[r["dtype"], str(r["loaded_requests"]), str(r["max_batch"]), str(r["identical"]),
             str(r["differed"]), "" if r["max_split_gap"] is None else f'{r["max_split_gap"]:g}']
            for r in d["invariance"]]  # fmt: skip


def md_table(head: list[str], rows: list[list[str]], right_from: int = 1) -> list[str]:
    align = ["---" if i < right_from else "---:" for i in range(len(head))]
    return ["| " + " | ".join(head) + " |", "| " + " | ".join(align) + " |",
            *["| " + " | ".join(r) + " |" for r in rows], ""]  # fmt: skip


def html_table(head: list[str], rows: list[list[str]]) -> str:
    th = "".join(f"<th>{esc(h)}</th>" for h in head)
    body = "".join("<tr>" + "".join(f"<td>{esc(c)}</td>" for c in r) + "</tr>" for r in rows)
    return f"<div class='scroll'><table><thead><tr>{th}</tr></thead>{body}</table></div>"


# --- RESULTS.md -------------------------------------------------------------------------------


def markdown(d: dict, chart_files: dict[str, str]) -> str:
    env = d["env"]
    icon = {"pass": "✅", "fail": "❌", "pending": "⏳"}
    lines = [
        "# Results",
        "",
        f"Measured on **{env.get('gpu', 'an NVIDIA GPU')}** ({env.get('gpu_memory_gib', '?')} GiB), "
        f"PyTorch {env.get('torch', '?')}, CUDA {env.get('cuda', '?')}, {env.get('date', '')}. "
        "Every number here comes from a JSON file in [`bench/results/`](bench/results), "
        "produced by `python bench/run_all.py` and drawn by `python bench/report.py`.",
        "",
        "A browsable version of this page: [`docs/results.html`](docs/results.html).",
        "",
        "## Gates",
        "",
        "| Check | Result | Detail |",
        "| --- | --- | --- |",
    ]
    for name, state, detail in gate_status(d):
        lines.append(f"| {name} | {icon[state]} {state} | {detail} |")
    tiles = headline(d)
    if tiles:
        lines += ["", "## Headline numbers", "", "| | | |", "| --- | --- | --- |"]
        lines += [f"| {k} | **{v}** | {s} |" for k, v, s in tiles]

    def img(key: str, alt: str) -> list[str]:
        return [f"![{alt}](docs/results/{chart_files[key]})", ""] if key in chart_files else []

    lines += ["", "## Correctness: identical output to Hugging Face", "",
              "Each square is one of the 200 test prompts. The engine and Hugging Face both "
              "generate 128 tokens greedily; green means every token matched. In BF16, amber "
              "marks a prompt where the two picked different tokens at a point where the "
              "model's top two choices were within 0.01 of each other (a coin flip that "
              "rounding can decide); anything else would be red.", ""]  # fmt: skip
    lines += img("gate1-1b", "Gate 1, 1B") + img("gate1-3b", "Gate 1, 3B")
    for title, head, rows, note in gate1_failures(d):
        lines += [
            f"**{title}.** Where the two implementations first chose a different "
            "token, and how far apart Hugging Face's top two logits were there.",
            "",
        ]
        lines += md_table(head, rows)
        lines += [note, ""] if note else []

    lines += ["## One user: how fast does a single reply stream?", "",
              "Each bar adds one technique on top of the one above it. Batch size 1, a "
              "128-token prompt, 256 generated tokens, greedy, median of three runs.", ""]  # fmt: skip
    for m in (M1, M3):
        lines += img(f"decode-{m}", f"decode speed {m}")
        rows = d["decode"][m]
        if rows:
            lines += ["| Variant | Tokens/s | ms per token | Peak GPU memory |",
                      "| --- | ---: | ---: | ---: |"]  # fmt: skip
            lines += [f'| {variant_label(r)} | {r["decode_tokens_per_s"]} | {r["itl_ms"]} | '
                      f'{r["peak_memory_gib"]} GiB |' for r in rows]  # fmt: skip
            lines.append("")

    lines += ["## Many users: what happens as load grows?", "",
              "The real server (`engine serve`, 1B, BF16) under N clients that each stream "
              "chat completions back to back for 60 seconds, 128 tokens per reply.", ""]  # fmt: skip
    lines += img("load-throughput", "throughput") + img("load-latency", "latency")
    if d["load"]:
        lines += ["| Users | Output tok/s | Requests/s | TTFT p50 | TTFT p99 | ITL p50 | ITL p99 "
                  "| Errors | Preemptions | Repeat answers differed |",
                  "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]  # fmt: skip
        for r in d["load"]:
            lines.append(
                f"| {r['concurrency']} | {r['output_tokens_per_s']:,.0f} | {r['requests_per_s']} "
                f"| {r['ttft_s']['p50'] * 1000:.0f} ms | {r['ttft_s']['p99'] * 1000:.0f} ms "
                f"| {r['itl_s']['p50'] * 1000:.1f} ms | {r['itl_s']['p99'] * 1000:.1f} ms "
                f"| {sum(r['errors'].values())} | {r['preemptions']:.0f} | {mismatch_note(r)} |"
            )
        lines.append("")
    if d["soak"]:
        s = d["soak"]
        lines += [f'**Soak (Gate 2):** {s["concurrency"]} streams for {s["duration_s"] / 60:.0f} '
                  f'minutes, {s["requests_ok"]:,} requests, {s["output_tokens_per_s"]:,.0f} '
                  f'tokens/s, {sum(s["errors"].values())} errors, '
                  f'repeat answers differed: {mismatch_note(s)}, KV memory '
                  f'in use at the end: {s["final_kv_used"]:.0%}.', ""]  # fmt: skip

    if d["invariance"]:
        lines += ["## Batching changes rounding, not math", "", INV_INTRO, ""]
        lines += md_table(INV_HEAD, inv_rows(d))
    lines += ["## Quality: does quantization hurt?", "",
              "Perplexity on the WikiText-2 test set (about 290,000 tokens, 2,048-token "
              "windows). Lower is better; the gap to BF16 is the price of smaller weights.", ""]  # fmt: skip
    lines += img("perplexity", "perplexity")
    if ppl_rows(d):
        lines += md_table(PPL_HEAD, ppl_rows(d), right_from=2)

    pre = [(m, d["prefill"][m]) for m in (M1, M3) if d["prefill"][m]]
    if pre:
        lines += ["## Reading prompts (prefill)", "",
                  "| Model | 512-token prompt | 2,000-token prompt | Peak memory |",
                  "| --- | ---: | ---: | ---: |"]  # fmt: skip
        for m, r in pre:
            p = r["prefill_tokens_per_s"]
            lines.append(f'| {SHORT[m]} | {p["512"]:,} tok/s | {p["2000"]:,} tok/s | '
                         f'{r["peak_memory_gib"]} GiB |')  # fmt: skip
        lines.append("")

    lines += ["## Reproduce", "", "```bash", "python bench/run_all.py      # about two hours",
              "python bench/report.py       # rebuilds this page", "```", ""]  # fmt: skip
    return "\n".join(lines)


# --- docs/results.html ----------------------------------------------------------------------

CSS = """
:root {
  /* Layout: a single reading column; charts full width; gate status first. */
  --bg: #f6f7f9; --panel: #ffffff; --fg: #15191f; --muted: #5b6472; --line: #dfe3e8;
  --grid: #c9cfd8; --dim: #b7c3d3; --accent: #2563eb; --accent2: #c27c0e;
  --good: #1f8a3b; --warn: #c27c0e; --bad: #c9372c;
  --font-display: "Archivo", system-ui, sans-serif;
  --font-body: "IBM Plex Sans", system-ui, sans-serif;
  --font-data: "IBM Plex Mono", ui-monospace, monospace;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #0e1116; --panel: #151a21; --fg: #e8ecf1; --muted: #98a2b3; --line: #262d38;
  --grid: #2c3440; --dim: #3a4658; --accent: #6ea8fe; --accent2: #f0b44c;
  --good: #3fb950; --warn: #e3a52f; --bad: #f2665c; color-scheme: dark; } }
:root[data-theme="dark"] {
  --bg: #0e1116; --panel: #151a21; --fg: #e8ecf1; --muted: #98a2b3; --line: #262d38;
  --grid: #2c3440; --dim: #3a4658; --accent: #6ea8fe; --accent2: #f0b44c;
  --good: #3fb950; --warn: #e3a52f; --bad: #f2665c; color-scheme: dark; }
body { background: var(--bg); color: var(--fg); font: 15px/1.55 var(--font-body); margin: 0; }
.wrap { max-width: 980px; margin: 0 auto; padding-block: 40px 64px;
  padding-left: max(16px, env(safe-area-inset-left, 0px));
  padding-right: max(16px, env(safe-area-inset-right, 0px)); }
h1 { font: 700 34px/1.15 var(--font-display); margin: 0 0 8px; text-wrap: balance; letter-spacing: -.01em; }
h2 { font: 650 21px/1.3 var(--font-display); margin: 48px 0 6px; text-wrap: balance; }
p { max-width: 68ch; color: var(--muted); margin: 0 0 16px; }
.lede { font-size: 16px; }
.gates { display: grid; grid-template-columns: repeat(auto-fit, minmax(270px, 1fr)); gap: 10px; margin-top: 24px; }
.gate { background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 12px 14px;
  display: grid; grid-template-columns: auto 1fr; gap: 2px 10px; align-items: center; }
.gate b { font-weight: 600; }
.gate span.d { grid-column: 2; color: var(--muted); font-size: 13px; font-family: var(--font-data); }
.pill { font: 600 11px/1 var(--font-data); text-transform: uppercase; letter-spacing: .06em;
  padding: 5px 7px; border-radius: 5px; grid-row: span 2; }
.pill.pass { color: var(--good); background: color-mix(in srgb, var(--good) 14%, transparent); }
.pill.fail { color: var(--bad); background: color-mix(in srgb, var(--bad) 14%, transparent); }
.pill.pending { color: var(--muted); background: color-mix(in srgb, var(--muted) 14%, transparent); }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 0;
  border-block: 1px solid var(--line); margin-top: 28px; }
.tile { padding: 18px 16px 18px 0; }
.tile .k { font: 600 11px/1 var(--font-data); letter-spacing: .08em; text-transform: uppercase; color: var(--muted); }
.tile .v { font: 700 30px/1.2 var(--font-display); margin-top: 8px; font-variant-numeric: tabular-nums; }
.tile .s { color: var(--muted); font-size: 13px; }
figure { margin: 20px 0; overflow-x: auto; }
figure svg { max-width: 100%; height: auto; display: block; }
.scroll { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font: 13px/1.4 var(--font-data); font-variant-numeric: tabular-nums; margin: 8px 0 16px; }
th, td { padding: 7px 10px; border-bottom: 1px solid var(--line); text-align: right; white-space: nowrap; }
th:first-child, td:first-child { text-align: left; }
th { color: var(--muted); font-weight: 500; }
code, pre { font-family: var(--font-data); font-size: 13px; }
pre { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 12px 14px; overflow-x: auto; }
.empty { color: var(--muted); font-style: italic; }
footer { margin-top: 56px; color: var(--muted); font-size: 13px; }
"""


def page(d: dict, svgs: dict[str, str], fragment: bool) -> str:
    env = d["env"]
    parts = []
    head = ('<title>engine benchmark report</title>\n'
            '<meta name="claude:viewport-fit-sides" content="cover">\n'
            '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@600;700'
            '&family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;600&display=swap">\n'
            f"<style>{CSS}</style>\n")  # fmt: skip
    parts.append('<div class="wrap">')
    parts.append("<h1>engine: benchmark report</h1>")
    parts.append(
        f'<p class="lede">A from-scratch LLM inference server, measured on '
        f"<b>{esc(env.get('gpu', 'an NVIDIA GPU'))}</b> "
        f"({esc(env.get('gpu_memory_gib', '?'))} GiB) with PyTorch {esc(env.get('torch', '?'))}, "
        f"CUDA {esc(env.get('cuda', '?'))}, on {esc(env.get('date', ''))}. Every number on this "
        f"page comes from a JSON file in <code>bench/results/</code>.</p>"
    )
    parts.append('<div class="gates">')
    for name, state, detail in gate_status(d):
        parts.append(f'<div class="gate"><span class="pill {state}">{state}</span>'
                     f'<b>{esc(name)}</b><span class="d">{esc(detail)}</span></div>')  # fmt: skip
    parts.append("</div>")
    tiles = headline(d)
    if tiles:
        parts.append('<div class="tiles">' + "".join(
            f'<div class="tile"><div class="k">{esc(k)}</div><div class="v">{esc(v)}</div>'
            f'<div class="s">{esc(s)}</div></div>' for k, v, s in tiles) + "</div>")  # fmt: skip

    def fig(key: str) -> str:
        return f"<figure>{svgs[key]}</figure>" if key in svgs else ""

    def empty(what: str) -> str:
        return f'<p class="empty">{esc(what)}: not run yet.</p>'

    parts.append("<h2>Correctness: identical output to Hugging Face</h2>")
    parts.append("<p>Each square is one of the 200 test prompts. The engine and Hugging Face both "
                 "generate 128 tokens greedily; green means every token matched. In BF16, amber "
                 "marks a prompt where the two picked different tokens at a point where the model's "
                 "top two choices were within 0.01 of each other, a coin flip rounding can decide. "
                 "Hover a square for the prompt.</p>")  # fmt: skip
    parts.append(fig("gate1-1b") or empty("Gate 1, 1B"))
    parts.append(fig("gate1-3b") or empty("Gate 1, 3B"))
    for title, cols, rows, note in gate1_failures(d):
        parts.append(f"<h3>{esc(title)}</h3>")
        parts.append(html_table(cols, rows))
        if note:
            parts.append(f"<p>{esc(note)}</p>")

    parts.append("<h2>One user: how fast does a single reply stream?</h2>")
    parts.append("<p>Each bar adds one technique on top of the one above it. Batch size 1, a "
                 "128-token prompt, 256 generated tokens, greedy, median of three runs. "
                 "Highlighted: the fastest.</p>")  # fmt: skip
    for m in (M1, M3):
        parts.append(fig(f"decode-{m}") or empty(f"Decode speed, {SHORT[m]}"))

    parts.append("<h2>Many users: what happens as load grows?</h2>")
    parts.append("<p>The real server (<code>engine serve</code>, 1B, BF16) under N clients that "
                 "each stream chat completions back to back for 60 seconds.</p>")  # fmt: skip
    parts.append(fig("load-throughput") or empty("Load sweep"))
    parts.append(fig("load-latency"))
    if d["load"]:
        rows = "".join(
            f'<tr><td>{r["concurrency"]}</td><td>{r["output_tokens_per_s"]:,.0f}</td>'
            f'<td>{r["requests_per_s"]}</td><td>{r["ttft_s"]["p50"] * 1000:.0f} ms</td>'
            f'<td>{r["ttft_s"]["p99"] * 1000:.0f} ms</td><td>{r["itl_s"]["p50"] * 1000:.1f} ms</td>'
            f'<td>{r["itl_s"]["p99"] * 1000:.1f} ms</td><td>{sum(r["errors"].values())}</td>'
            f'<td>{r["preemptions"]:.0f}</td><td>{esc(mismatch_note(r))}</td></tr>'
            for r in d["load"])  # fmt: skip
        parts.append('<div class="scroll"><table><thead><tr><th>users</th><th>output tok/s</th>'
                     "<th>requests/s</th><th>TTFT p50</th><th>TTFT p99</th><th>ITL p50</th>"
                     f"<th>ITL p99</th><th>errors</th><th>preemptions</th>"
                     f"<th>repeat answers differed</th></tr></thead>{rows}"
                     "</table></div>")  # fmt: skip
    if d["soak"]:
        s = d["soak"]
        parts.append(f'<p><b>Soak (Gate 2):</b> {s["concurrency"]} streams for '
                     f'{s["duration_s"] / 60:.0f} minutes, {s["requests_ok"]:,} requests, '
                     f'{s["output_tokens_per_s"]:,.0f} tokens/s, {sum(s["errors"].values())} errors, '
                     f'repeat answers differed: {esc(mismatch_note(s))}, KV memory '
                     f'in use at the end: {s["final_kv_used"]:.0%}.</p>')  # fmt: skip

    if d["invariance"]:
        parts.append("<h2>Batching changes rounding, not math</h2>")
        parts.append(f"<p>{esc(INV_INTRO)}</p>")
        parts.append(html_table(INV_HEAD, inv_rows(d)))
    parts.append("<h2>Quality: does quantization hurt?</h2>")
    parts.append("<p>Perplexity on the WikiText-2 test set, 2,048-token windows. Lower is better; "
                 "the gap to BF16 is the price of smaller weights.</p>")  # fmt: skip
    parts.append(fig("perplexity") or empty("Perplexity"))
    if ppl_rows(d):
        parts.append(html_table(PPL_HEAD, ppl_rows(d)))

    pre = [(m, d["prefill"][m]) for m in (M1, M3) if d["prefill"][m]]
    if pre:
        parts.append("<h2>Reading prompts (prefill)</h2>")
        rows = "".join(
            f'<tr><td>{SHORT[m]}</td><td>{r["prefill_tokens_per_s"]["512"]:,}</td>'
            f'<td>{r["prefill_tokens_per_s"]["2000"]:,}</td><td>{r["peak_memory_gib"]} GiB</td></tr>'
            for m, r in pre)  # fmt: skip
        parts.append("<div class='scroll'><table><thead><tr><th>model</th><th>512-token prompt, "
                     "tok/s</th><th>2,000-token prompt, tok/s</th><th>peak memory</th></tr>"
                     f"</thead>{rows}</table></div>")  # fmt: skip

    parts.append("<h2>Reproduce</h2><pre>python bench/run_all.py      # about two hours\n"
                 "python bench/report.py       # rebuilds this page</pre>")  # fmt: skip
    parts.append(f"<footer>{esc(env.get('os', ''))} · Python {esc(env.get('python', ''))}</footer>")
    parts.append("</div>")
    body = head + "\n".join(parts)
    if fragment:
        return body
    return ('<!doctype html>\n<html lang="en"><head><meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
            f"{head}</head><body>\n" + "\n".join(parts) + "\n</body></html>\n")  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fragment", type=Path, help="also write the page body only (for embedding)")
    args = ap.parse_args()
    d = collect()
    CHARTS.mkdir(parents=True, exist_ok=True)
    files = {}
    for key, s in charts(d, FIXED).items():
        name = f"{key}.svg"
        (CHARTS / name).write_text(s, encoding="utf-8")
        files[key] = name
    (ROOT / "RESULTS.md").write_text(markdown(d, files), encoding="utf-8")
    themed = charts(d, TOKENS)
    (DOCS / "results.html").write_text(page(d, themed, fragment=False), encoding="utf-8")
    if args.fragment:
        args.fragment.write_text(page(d, themed, fragment=True), encoding="utf-8")
    print(f"wrote RESULTS.md, docs/results.html and {len(files)} charts")


if __name__ == "__main__":
    main()

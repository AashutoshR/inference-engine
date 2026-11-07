"""Run every benchmark and gate, one after another, and save all results.

    python bench/run_all.py                      # everything, about two hours on 8 GB
    python bench/run_all.py --only decode ppl    # some groups
    python bench/run_all.py --soak-seconds 60    # short soak instead of Gate 2's 30 minutes

Groups, in order: decode, prefill, ppl, load, tests, gate1, invariance, soak. Each step runs in its
own process (so GPU memory is fully released between steps), writes JSON to
bench/results/, and a failure is logged without stopping the rest. Progress goes to
bench/results/run_all.log. Afterwards: python bench/report.py
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "bench" / "results"
PY = sys.executable
M1 = "models/Llama-3.2-1B-Instruct"
M3 = "models/Llama-3.2-3B-Instruct"
LOG = RESULTS / "run_all.log"


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def run(name: str, args: list[str], env: dict | None = None, timeout: float = 7200) -> bool:
    log(f"START {name}")
    start = time.time()
    try:
        proc = subprocess.run(
            args, cwd=ROOT, env={**os.environ, **(env or {})}, timeout=timeout,
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )  # fmt: skip
        tail = "\n".join((proc.stdout + proc.stderr).strip().splitlines()[-4:])
        ok = proc.returncode == 0
    except subprocess.TimeoutExpired:
        ok, tail = False, "timed out"
    log(f"{'DONE ' if ok else 'FAIL '} {name} ({time.time() - start:.0f}s)\n    {tail}")
    return ok


def decode() -> None:
    for model, variants in [
        (M1, [("static", "none"), ("eager", "none"), ("triton", "none"), ("graphs", "none"),
              ("graphs", "int4"), ("graphs", "fp8")]),
        (M3, [("static", "none"), ("graphs", "none"), ("graphs", "int4"), ("graphs", "fp8")]),
    ]:  # fmt: skip
        for mode, quant in variants:
            run(
                f"decode {Path(model).name} {mode} {quant}",
                [PY, "bench/bench_decode.py", "--model", model, "--mode", mode, "--quant", quant],
            )
    run(
        f"decode {Path(M1).name} spec ngram",
        [PY, "bench/bench_decode.py", "--model", M1, "--mode", "spec", "--spec", "ngram"],
    )
    run(
        f"decode {Path(M3).name} spec ngram",
        [PY, "bench/bench_decode.py", "--model", M3, "--mode", "spec", "--spec", "ngram"],
    )
    # 3B target + 1B draft only fit together in INT4.
    run(f"decode {Path(M3).name} spec draft-1B int4",
        [PY, "bench/bench_decode.py", "--model", M3, "--mode", "spec", "--spec", M1,
         "--quant", "int4"])  # fmt: skip


def prefill() -> None:
    for model in (M1, M3):
        run(f"prefill {Path(model).name}", [PY, "bench/bench_baseline.py", "--model", model])


def ppl() -> None:
    for model in (M1, M3):
        for quant in ("none", "int4", "fp8"):
            run(
                f"perplexity {Path(model).name} {quant}",
                [PY, "bench/perplexity.py", "--model", model, "--quant", quant],
            )


class Server:
    def __init__(self, model: str, port: int = 8011, extra: list[str] | None = None) -> None:
        self.url = f"http://127.0.0.1:{port}"
        self.log = (RESULTS / "server.log").open("a", encoding="utf-8")
        self.proc = subprocess.Popen(
            [PY, "-m", "engine.cli", "serve", "--model", model, "--port", str(port),
             *(extra or [])],
            cwd=ROOT, stdout=self.log, stderr=subprocess.STDOUT,
        )  # fmt: skip

    def __enter__(self) -> Server:
        for _ in range(240):
            try:
                urllib.request.urlopen(self.url + "/health", timeout=2)
                return self
            except Exception:
                if self.proc.poll() is not None:
                    raise RuntimeError("server exited; see bench/results/server.log") from None
                time.sleep(1)
        raise RuntimeError("server did not become healthy")

    def __exit__(self, *exc) -> None:
        self.proc.terminate()
        try:
            self.proc.wait(30)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        self.log.close()


def load() -> None:
    try:
        with Server(M1) as srv:
            for c in (1, 4, 16, 32, 64):
                run(f"load {Path(M1).name} c{c}",
                    [PY, "bench/load_test.py", "--url", srv.url, "--concurrency", str(c),
                     "--duration", "60", "--report-every", "30"])  # fmt: skip
    except RuntimeError as exc:
        log(f"FAIL load: {exc}")


def soak(seconds: float) -> None:
    try:
        with Server(M1) as srv:
            run("soak (Gate 2)",
                [PY, "bench/load_test.py", "--url", srv.url, "--concurrency", "64",
                 "--duration", str(seconds), "--report-every", "60", "--name", "gate2-soak"],
                timeout=seconds + 900)  # fmt: skip
    except RuntimeError as exc:
        log(f"FAIL soak: {exc}")


def tests() -> None:
    for name, marker in (("cpu", "not gpu and not slow"), ("gpu", "gpu and not slow")):
        xml = RESULTS / f"tests-{name}.xml"
        run(f"tests {name}", [PY, "-m", "pytest", "-q", "-m", marker, "-p", "no:cacheprovider",
                              f"--junitxml={xml}"])  # fmt: skip


def gate1() -> None:
    run("gate1 1B fp32", [PY, "-m", "pytest", "-q", "-m", "slow", "tests/test_golden.py",
                          "-p", "no:cacheprovider"], env={"ENGINE_MODEL_DIR": M1})  # fmt: skip
    run("gate1 3B bf16", [PY, "-m", "pytest", "-q", "-m", "slow", "tests/test_golden.py",
                          "-p", "no:cacheprovider"],
        env={"ENGINE_MODEL_DIR": M3, "ENGINE_GOLDEN_DTYPE": "bfloat16"})  # fmt: skip


GROUPS = ["decode", "prefill", "ppl", "load", "tests", "gate1", "invariance", "soak"]


def invariance() -> None:
    for dtype in ("float32", "bfloat16"):
        run(f"invariance 1B {dtype}",
            [PY, "bench/batch_invariance.py", "--model", M1, "--dtype", dtype])  # fmt: skip


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--only",
        nargs="*",
        default=None,
        choices=GROUPS,
    )
    p.add_argument("--soak-seconds", type=float, default=1800)
    args = p.parse_args()
    RESULTS.mkdir(exist_ok=True)
    groups = args.only or GROUPS
    log(f"run_all: {' '.join(groups)}")
    steps = {"decode": decode, "prefill": prefill, "ppl": ppl, "load": load, "tests": tests,
             "gate1": gate1, "invariance": invariance,
             "soak": lambda: soak(args.soak_seconds)}  # fmt: skip
    for g in groups:
        steps[g]()
    (RESULTS / "environment.json").write_text(json.dumps(environment(), indent=2) + "\n")
    log("run_all: finished")


def environment() -> dict:
    import platform

    import torch

    return {
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "gpu_memory_gib": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1)
        if torch.cuda.is_available() else None,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "python": platform.python_version(),
        "os": platform.platform(),
        "date": time.strftime("%Y-%m-%d"),
    }  # fmt: skip


if __name__ == "__main__":
    main()

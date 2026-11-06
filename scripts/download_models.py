"""Download model weights into ./models.

Meta's Llama repos are gated: accept the license on each model's Hugging Face page,
then run `hf auth login` once before this script. `--source unsloth` downloads
unsloth's ungated copies instead (same weights, tokenizer and chat template; the
Llama license still applies to you).

    python scripts/download_models.py                    # 1B only (fits an 8 GB GPU)
    python scripts/download_models.py --also 3b          # plus 3B (about 6.4 GB in bf16)
    python scripts/download_models.py --source unsloth   # no Hugging Face login needed
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

MODELS = {
    "1b": "Llama-3.2-1B-Instruct",
    "3b": "Llama-3.2-3B-Instruct",
    "8b": "Llama-3.1-8B-Instruct",  # 16 GB in bf16: needs a 24 GB GPU or INT4
}
SOURCES = {"meta": "meta-llama", "unsloth": "unsloth"}
# Top-level files only (ignore_patterns below): Meta's repos also ship original/
# checkpoints we never load.
PATTERNS = ["*.json", "*.safetensors"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--also", nargs="*", default=[], choices=sorted(MODELS))
    parser.add_argument("--source", default="meta", choices=sorted(SOURCES))
    args = parser.parse_args()

    for key in dict.fromkeys(["1b", *args.also]):
        name = MODELS[key]
        repo = f"{SOURCES[args.source]}/{name}"
        dest = Path("models") / name
        print(f"downloading {repo} -> {dest}")
        snapshot_download(repo, local_dir=dest, allow_patterns=PATTERNS, ignore_patterns=["*/*"])


if __name__ == "__main__":
    main()

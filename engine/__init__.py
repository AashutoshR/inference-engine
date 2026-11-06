"""An LLM inference engine built from scratch.

Importing this package must stay cheap and must not import torch, so the CLI's
``info`` command and the config tests run without a GPU stack installed.
"""

__version__ = "0.1.0"

#!/usr/bin/env python3
"""Start the live UI.

  python scripts/serve_ui.py                                     # oracle (rule-coded agent)
  python scripts/serve_ui.py --policy openai --model qwen3:1.7b  # local model via Ollama/vLLM
  python scripts/serve_ui.py --policy hf --model Qwen/Qwen3-1.7B --adapter checkpoints/lora
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import random

from homesim.ui.server import serve


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--policy", choices=["oracle", "openai", "hf"], default="oracle")
    ap.add_argument("--model", default="qwen3:1.7b")
    ap.add_argument("--base-url", default="http://localhost:11434/v1")
    ap.add_argument("--adapter", default=None)
    args = ap.parse_args()
    seed = args.seed if args.seed is not None else random.randrange(1 << 20)
    serve(args.host, args.port, seed, args.policy, args.model, args.base_url, args.adapter)


if __name__ == "__main__":
    main()

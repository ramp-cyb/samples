#!/usr/bin/env python3
"""Evaluate a policy on a dataset split and print the metric table.

  python scripts/run_eval.py --policy oracle --data data/eval_in_dist.jsonl
  python scripts/run_eval.py --policy hf --model Qwen/Qwen3-1.7B --data data/eval_in_dist.jsonl
  python scripts/run_eval.py --policy hf --model Qwen/Qwen3-1.7B --adapter checkpoints/lora --data ...
  python scripts/run_eval.py --policy openai --base-url http://localhost:11434/v1 --model qwen3:1.7b --data ...
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json

from homesim.datagen import read_jsonl
from homesim.evaluate import evaluate_policy, format_summary, summarize


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--policy", choices=["oracle", "hf", "openai"], default="oracle")
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--base-url", default="http://localhost:11434/v1")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-turns", type=int, default=20)
    ap.add_argument("--out", default=None, help="write per-episode rows as JSONL")
    ap.add_argument("--label", default=None)
    args = ap.parse_args()

    records = read_jsonl(args.data)
    if args.limit:
        records = records[: args.limit]

    if args.policy == "oracle":
        from homesim.intents import Scenario
        from homesim.oracle import OraclePolicy, plan
        from homesim.world import generate_house

        def factory(rec):
            house = generate_house(rec["house_seed"])
            return OraclePolicy(plan(house, Scenario(**rec["scenario"])))
    elif args.policy == "hf":
        from homesim.agents.hf_policy import HFPolicy

        shared = HFPolicy(args.model, adapter=args.adapter)
        factory = lambda rec: shared
    else:
        from homesim.agents.openai_compat import OpenAICompatPolicy

        shared = OpenAICompatPolicy(base_url=args.base_url, model=args.model)
        factory = lambda rec: shared

    done = {"n": 0}

    def progress(rec, tr, sc):
        done["n"] += 1
        if done["n"] % 10 == 0:
            print(f"  {done['n']}/{len(records)}", flush=True)

    rows = evaluate_policy(factory, records, max_turns=args.max_turns, on_episode=progress)
    label = args.label or (f"{args.policy}:{args.model}" + (f"+{os.path.basename(args.adapter)}" if args.adapter else ""))
    print(f"\n== {label} on {args.data} ({len(rows)} episodes) ==")
    print(format_summary(summarize(rows)))
    print("\nby intent:")
    print(format_summary(summarize(rows, by="intent")))
    if args.out:
        with open(args.out, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"\nper-episode rows -> {args.out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run several policies over the same eval splits and print one comparison table.

  python scripts/benchmark.py \
      --config '[{"label":"base-1.7b","policy":"openai","model":"qwen3:1.7b"},
                 {"label":"tuned-1.7b","policy":"hf","model":"Qwen/Qwen3-1.7B","adapter":"checkpoints/lora"}]'

Or point it at a JSON file with --config-file. The oracle is always included as
the upper bound unless --no-oracle is passed.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json

from homesim.datagen import read_jsonl
from homesim.evaluate import evaluate_policy, summarize

COLUMNS = ["success", "state_match", "terminal_correct", "tool_validity", "invalid_calls", "redundant_calls", "unsafe", "turns"]


def build_factory(spec: dict) -> callable:
    policy = spec["policy"]
    if policy == "oracle":
        from homesim.intents import Scenario
        from homesim.oracle import OraclePolicy, plan
        from homesim.world import generate_house

        def factory(rec):
            house = generate_house(rec["house_seed"])
            return OraclePolicy(plan(house, Scenario(**rec["scenario"])))

        return factory
    if policy == "hf":
        from homesim.agents.hf_policy import HFPolicy

        shared = HFPolicy(spec["model"], adapter=spec.get("adapter"))
        return lambda rec: shared
    from homesim.agents.openai_compat import OpenAICompatPolicy

    shared = OpenAICompatPolicy(base_url=spec.get("base_url", "http://localhost:11434/v1"), model=spec["model"])
    return lambda rec: shared


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--splits", nargs="*", default=["eval_in_dist", "eval_heldout_phrasing"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--config-file", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-oracle", action="store_true")
    ap.add_argument("--out", default="data/benchmark.json")
    args = ap.parse_args()

    specs = json.loads(args.config) if args.config else (json.load(open(args.config_file)) if args.config_file else [])
    if not args.no_oracle:
        specs = [{"label": "oracle (upper bound)", "policy": "oracle"}] + specs
    if not specs:
        ap.error("give at least one policy via --config or --config-file, or drop --no-oracle")

    results: dict[str, dict[str, dict[str, float]]] = {}
    for spec in specs:
        label = spec.get("label", spec.get("model", spec["policy"]))
        factory = build_factory(spec)
        results[label] = {}
        for split in args.splits:
            records = read_jsonl(os.path.join(args.data_dir, f"{split}.jsonl"))
            if args.limit:
                records = records[: args.limit]
            print(f"running {label} on {split} ({len(records)} episodes)...", flush=True)
            rows = evaluate_policy(factory, records, max_turns=20)
            results[label][split] = summarize(rows)["all"]

    width = max(len(l) for l in results) + 2
    for split in args.splits:
        print(f"\n== {split} ==")
        print(f"{'policy':{width}s} " + " ".join(f"{c:>16s}" for c in COLUMNS))
        for label, per_split in results.items():
            s = per_split[split]
            print(f"{label:{width}s} " + " ".join(f"{s[c]:>16.3f}" for c in COLUMNS))
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()

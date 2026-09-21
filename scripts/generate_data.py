#!/usr/bin/env python3
"""Generate train/eval datasets.

  python scripts/generate_data.py --train-houses 400 --per-house 6 --out data
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import collections
import json

from homesim.datagen import generate_split, write_jsonl


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--train-houses", type=int, default=400)
    ap.add_argument("--per-house", type=int, default=6)
    ap.add_argument("--eval-houses", type=int, default=60)
    ap.add_argument("--eval-per-house", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--batch-calls", action="store_true",
                    help="emit all state-changing calls of a step group in one assistant turn (shorter episodes)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    specs = [
        ("train", args.train_houses, args.per_house, args.seed),
        ("eval_in_dist", args.eval_houses, args.eval_per_house, args.seed + 1),
        ("eval_heldout_phrasing", args.eval_houses, args.eval_per_house, args.seed + 2),
    ]
    stats = {}
    for split, houses, per, seed in specs:
        path = os.path.join(args.out, f"{split}.jsonl")
        records = list(generate_split(split, houses, per, seed, batch=args.batch_calls))
        n = write_jsonl(path, records)
        by_intent = collections.Counter(r["scenario"]["intent"] for r in records)
        turns = sum(len([m for m in r["messages"] if m["role"] == "assistant"]) for r in records) / max(n, 1)
        stats[split] = {"records": n, "avg_assistant_turns": round(turns, 2), "intents": dict(by_intent.most_common())}
        print(f"{split:24s} {n:6d} records -> {path}  (avg {turns:.1f} assistant turns)")
    with open(os.path.join(args.out, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps({k: v["intents"] for k, v in stats.items()}, indent=2)[:600])


if __name__ == "__main__":
    main()

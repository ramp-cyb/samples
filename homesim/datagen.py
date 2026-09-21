"""Synthetic dataset generation in chat/tool-call format.

Splits:
  train                 training houses  x training templates
  eval_in_dist          held-out houses  x training templates
  eval_heldout_phrasing held-out houses  x held-out templates (unseen English)
"""

from __future__ import annotations

import json
import random
from typing import Any, Iterable

from homesim.episode import run_episode
from homesim.evaluate import make_expected
from homesim.intents import INTENTS, sample_scenario
from homesim.knowledge import KnowledgeBase
from homesim.oracle import OraclePolicy, plan
from homesim.tools import TOOL_SCHEMAS
from homesim.world import generate_house

HELDOUT_TEMPLATE_INDICES = {8, 9}  # last two templates of every intent are never trained on
EVAL_HOUSE_OFFSET = 1_000_000  # eval houses use seeds >= this so they never overlap train


def build_record(house_seed: int, scenario_rng_seed: int, split: str, intent: str | None = None,
                 template_index: int | None = None, batch: bool = False) -> dict[str, Any]:
    house = generate_house(house_seed)
    rng = random.Random(scenario_rng_seed)
    sc = sample_scenario(house, rng, intent, template_index)
    oracle_plan = plan(house, sc)
    expected = make_expected(house, sc, oracle_plan)
    kb = KnowledgeBase.for_house(house)
    tr = run_episode(house, kb, OraclePolicy(oracle_plan, batch=batch), sc.utterance, max_turns=60)
    assert tr.terminal == oracle_plan.expected_terminal
    return {
        "id": f"h{house_seed}-s{scenario_rng_seed}",
        "split": split,
        "house_seed": house_seed,
        "scenario": sc.to_dict(),
        "n_state_calls": len(oracle_plan.state_calls),
        "messages": tr.messages,
        "tools": TOOL_SCHEMAS,
        "expected": expected.to_dict(),
    }


def _template_choices(intent: str, heldout: bool) -> list[int]:
    n = len(INTENTS[intent].templates)
    if intent == "status_query":
        return [0]
    idx = [i for i in range(n) if (i in HELDOUT_TEMPLATE_INDICES) == heldout]
    return idx


def generate_split(split: str, n_houses: int, per_house: int, seed: int, batch: bool = False,
                   max_noop_fraction: float = 0.25) -> Iterable[dict[str, Any]]:
    """Yield records. Scenarios that require no state change are capped to a fraction of the split."""
    rng = random.Random(seed)
    heldout = split == "eval_heldout_phrasing"
    house_base = 0 if split == "train" else EVAL_HOUSE_OFFSET + (0 if split == "eval_in_dist" else 500_000)
    n_noop = 0
    total = 0
    for hi in range(n_houses):
        house_seed = house_base + seed * 10_000 + hi
        house = generate_house(house_seed)
        applicable = [d.name for d in INTENTS.values() if d.applies(house)]
        for k in range(per_house):
            intent = rng.choice(applicable)
            ti = rng.choice(_template_choices(intent, heldout))
            scen_seed = rng.randrange(1 << 30)
            rec = build_record(house_seed, scen_seed, split, intent, ti, batch=batch)
            noop = rec["n_state_calls"] == 0 and rec["expected"]["terminal"] == "done" and intent not in ("status_query", "no_action")
            if noop and total > 0 and n_noop / total >= max_noop_fraction:
                continue
            n_noop += int(noop)
            total += 1
            yield rec


def write_jsonl(path: str, records: Iterable[dict[str, Any]]) -> int:
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

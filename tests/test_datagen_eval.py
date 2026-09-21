import json

from homesim.datagen import HELDOUT_TEMPLATE_INDICES, generate_split, read_jsonl, write_jsonl
from homesim.episode import assistant_message
from homesim.evaluate import Expected, evaluate_policy, oracle_baseline, score, summarize
from homesim.world import generate_house


def test_records_are_json_serialisable_and_well_formed(tmp_path):
    recs = list(generate_split("train", 3, 4, seed=0))
    assert recs
    path = tmp_path / "d.jsonl"
    write_jsonl(str(path), recs)
    back = read_jsonl(str(path))
    assert back == json.loads(json.dumps(recs))
    for r in recs:
        roles = [m["role"] for m in r["messages"]]
        assert roles[:2] == ["system", "user"]
        assert roles[-1] == "tool"  # terminal call result
        last_assistant = [m for m in r["messages"] if m["role"] == "assistant"][-1]
        assert last_assistant["tool_calls"][0]["function"]["name"] == r["expected"]["terminal"]


def test_splits_use_disjoint_houses_and_templates():
    train = list(generate_split("train", 4, 5, seed=0))
    held = list(generate_split("eval_heldout_phrasing", 4, 5, seed=0))
    assert {r["house_seed"] for r in train}.isdisjoint({r["house_seed"] for r in held})
    for r in train:
        tid = r["scenario"]["template_id"]
        if not tid.startswith("status_query"):
            assert int(tid.split(":")[1]) not in HELDOUT_TEMPLATE_INDICES
    for r in held:
        tid = r["scenario"]["template_id"]
        if not tid.startswith("status_query"):
            assert int(tid.split(":")[1]) in HELDOUT_TEMPLATE_INDICES


def test_oracle_scores_perfectly():
    recs = list(generate_split("eval_in_dist", 4, 4, seed=3))
    rows = oracle_baseline(recs)
    s = summarize(rows)["all"]
    assert s["success"] == 1.0 and s["redundant_calls"] == 0.0 and s["invalid_calls"] == 0.0


def test_bad_policy_is_penalised():
    recs = [r for r in generate_split("eval_in_dist", 6, 3, seed=5) if r["n_state_calls"] > 0][:5]
    assert recs

    class LazyPolicy:
        """Says done immediately without doing anything."""

        def __call__(self, messages, tools):
            return assistant_message("Done.", [("done", {"message": "All sorted."})])

    rows = evaluate_policy(lambda rec: LazyPolicy(), recs)
    for r in rows:
        assert r["terminal_correct"] == (r["terminal"] == "done")
        assert r["state_match"] is False
        assert r["missing_changes"] > 0
        assert r["success"] is False


def test_text_only_reply_counts_as_parse_failure():
    recs = list(generate_split("eval_in_dist", 1, 1, seed=8))

    class Chatty:
        def __call__(self, messages, tools):
            return {"role": "assistant", "content": "Sure, I will do that now."}

    rows = evaluate_policy(lambda rec: Chatty(), recs, max_turns=3)
    assert rows[0]["parse_failures"] == 3 and rows[0]["terminal"] is None


def test_unsafe_unlock_is_flagged():
    for seed in range(200):
        h = generate_house(seed)
        if h.is_night:
            break
    from homesim.intents import sample_scenario
    import random
    from homesim.oracle import plan
    from homesim.evaluate import make_expected
    from homesim import KnowledgeBase, run_episode
    sc = sample_scenario(h, random.Random(0), "unlock_door", 0)
    expected = make_expected(h, sc)
    assert expected.terminal == "ask_user"

    class Reckless:
        def __init__(self):
            self.n = 0

        def __call__(self, messages, tools):
            self.n += 1
            if self.n == 1:
                return assistant_message("Unlocking.", [("set_device", {"device_id": sc.params["door_id"], "attribute": "locked", "value": False})])
            return assistant_message("Done.", [("done", {"message": "Unlocked."})])

    initial = h.device_states()
    tr = run_episode(h, KnowledgeBase.for_house(h), Reckless(), sc.utterance)
    s = score(tr, expected, initial)
    assert s.unsafe and not s.success

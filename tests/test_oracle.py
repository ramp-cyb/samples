import random

import pytest

from homesim import KnowledgeBase, generate_house, run_episode
from homesim.intents import INTENTS, Scenario, sample_scenario
from homesim.knowledge import generate_profile
from homesim.oracle import OraclePolicy, plan


def _run(seed, intent, ti=0, batch=False):
    h = generate_house(seed)
    sc = sample_scenario(h, random.Random(seed), intent, ti)
    pl = plan(h, sc)
    tr = run_episode(h, KnowledgeBase.for_house(h), OraclePolicy(pl, batch=batch), sc.utterance, max_turns=60)
    return h, sc, pl, tr


@pytest.mark.parametrize("seed", range(25))
@pytest.mark.parametrize("intent", sorted(INTENTS))
def test_every_intent_runs_clean(seed, intent):
    h0 = generate_house(seed)
    if not INTENTS[intent].applies(h0):
        pytest.skip("intent not applicable to this house")
    h, sc, pl, tr = _run(seed, intent)
    assert tr.terminal == pl.expected_terminal
    assert all(r.ok for r in tr.tool_results)
    assert all(r.changed for r in tr.tool_results if r.name in ("set_device", "create_schedule", "add_to_list"))


def test_movie_mode_uses_profile_values():
    seed = 7
    h, sc, pl, tr = _run(seed, "movie_mode")
    p = generate_profile(seed)
    for d in h.devices_of_kind("light", "living_room"):
        assert d.state["brightness"] == p.movie_brightness and d.state["color_temp"] == "warm"
    assert h.devices["living_room.blinds"].state["position"] == "closed"
    assert h.devices["living_room.tv"].state["power"] == "on"
    assert h.devices["living_room.speaker"].state["volume"] == p.movie_volume


def test_bedtime_locks_and_lights():
    h, sc, pl, tr = _run(3, "bedtime")
    assert all(d.state["power"] == "off" for d in h.devices_of_kind("light"))
    assert all(d.state["locked"] for d in h.devices_of_kind("lock"))
    assert h.thermostat().state == {"mode": "heat", "target": generate_profile(3).night_temp}


def test_unlock_at_night_asks_instead():
    found = False
    for seed in range(100):
        h = generate_house(seed)
        if h.is_night:
            found = True
            h, sc, pl, tr = _run(seed, "unlock_door")
            assert tr.terminal == "ask_user"
            assert h.devices[sc.params["door_id"]].state["locked"] == generate_house(seed).devices[sc.params["door_id"]].state["locked"]
            break
    assert found


def test_unlock_by_day_unlocks():
    for seed in range(100):
        h = generate_house(seed)
        if not h.is_night:
            h, sc, pl, tr = _run(seed, "unlock_door")
            assert tr.terminal == "done"
            assert h.devices[sc.params["door_id"]].state["locked"] is False
            return
    raise AssertionError("no daytime house found")


def test_ambiguous_room_asks():
    for seed in range(200):
        h = generate_house(seed)
        if h.user_location is None:
            sc = sample_scenario(h, random.Random(0), "too_dark", 0)  # template 0 has no {room}
            assert sc.ambiguous_room
            pl = plan(h, sc)
            assert pl.expected_terminal == "ask_user" and len(pl.steps) == 1
            return
    raise AssertionError("no house without user location found")


def test_room_resolves_from_user_location():
    for seed in range(200):
        h = generate_house(seed)
        if h.user_location is not None:
            sc = sample_scenario(h, random.Random(0), "too_dark", 0)
            assert sc.room == h.user_location and not sc.ambiguous_room
            return


def test_too_cold_rule():
    seed = 11
    h = generate_house(seed)
    p = generate_profile(seed)
    h.thermostat().state["target"] = p.comfort_temp - 3
    sc = Scenario("too_cold", {"room": "kitchen"}, "too_cold:0", "I'm cold.")
    pl = plan(h, sc)
    targets = [s.arguments["value"] for s in pl.state_calls if s.arguments.get("attribute") == "target"]
    assert targets == [p.comfort_temp]
    h.thermostat().state["target"] = p.comfort_temp
    pl = plan(h, sc)
    targets = [s.arguments["value"] for s in pl.state_calls if s.arguments.get("attribute") == "target"]
    assert targets == [p.comfort_temp + 2]


def test_kids_room_cap_after_20():
    for seed in range(300):
        h = generate_house(seed)
        if "kids_room" in h.rooms and h.hour >= 20:
            p = generate_profile(seed)
            sc = Scenario("too_dark", {"room": "kids_room"}, "too_dark:1", "It's too dark in the kids' room.", room="kids_room")
            pl = plan(h, sc)
            levels = {s.arguments["value"] for s in pl.state_calls if s.arguments.get("attribute") == "brightness"}
            assert levels <= {min(p.reading_light, p.kids_light_cap)}
            return
    raise AssertionError("no matching house")


def test_batch_and_stepwise_reach_same_state():
    for intent in ("bedtime", "movie_mode", "leaving_home"):
        h1, _, _, _ = _run(4, intent, batch=False)
        h2, _, _, tr2 = _run(4, intent, batch=True)
        assert h1.device_states() == h2.device_states()
        assert tr2.turns <= 3


def test_utterances_avoid_tool_vocabulary():
    banned = ("set_device", "device_id", "search_context", "add_to_list")
    for d in INTENTS.values():
        for t in d.templates:
            assert not any(b in t.text for b in banned)

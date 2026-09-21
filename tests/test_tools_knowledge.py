from homesim import KnowledgeBase, ToolExecutor, generate_house
from homesim.knowledge import generate_profile
from homesim.tools import TOOL_SCHEMAS


def test_tool_schemas_are_well_formed():
    names = [t["function"]["name"] for t in TOOL_SCHEMAS]
    assert names == ["search_context", "get_state", "set_device", "create_schedule", "add_to_list", "ask_user", "done"]
    for t in TOOL_SCHEMAS:
        assert t["function"]["parameters"]["type"] == "object"


def test_retrieval_finds_right_documents():
    h = generate_house(5)
    kb = KnowledgeBase.for_house(h)
    assert kb.search("movie mode scene living room")[0].id == "scene_movie"
    assert kb.search("bedtime routine")[0].id == "scene_bedtime"
    assert "rule_night_lock" in [d.id for d in kb.search("unlock door rule")]
    assert kb.search("cold comfort temperature preference")[0].id == "pref_temperature"
    assert kb.search("guests coming to stay guest room")[0].id == "pref_guests"
    assert kb.search("xyzzy plugh") == []


def test_profile_values_appear_in_docs():
    h = generate_house(9)
    p = generate_profile(9)
    kb = KnowledgeBase.for_house(h, p)
    assert f"{p.movie_brightness}%" in kb.get("scene_movie").text
    assert f"{p.night_temp} degrees" in kb.get("scene_bedtime").text


def test_executor_errors_are_returned_not_raised():
    h = generate_house(5)
    ex = ToolExecutor(h, KnowledgeBase.for_house(h))
    r = ex.execute("set_device", {"device_id": "nope", "attribute": "power", "value": "on"})
    assert not r.ok and "unknown device_id" in r.result
    r = ex.execute("set_device", {"device_id": "living_room.tv"})
    assert not r.ok and "missing" in r.result
    r = ex.execute("teleport", {})
    assert not r.ok and "unknown tool" in r.result
    r = ex.execute("done", {"message": "ok"})
    assert r.ok and r.terminal


def test_executor_tracks_changes():
    h = generate_house(5)
    ex = ToolExecutor(h, KnowledgeBase.for_house(h))
    h.set_device("living_room.tv", "power", "off")
    assert ex.execute("set_device", {"device_id": "living_room.tv", "attribute": "power", "value": "off"}).changed is False
    assert ex.execute("set_device", {"device_id": "living_room.tv", "attribute": "power", "value": "on"}).changed is True

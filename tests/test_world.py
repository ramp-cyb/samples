import pytest

from homesim.world import WorldError, generate_house


def test_generation_is_deterministic():
    a, b = generate_house(42), generate_house(42)
    assert a.snapshot() == b.snapshot()
    assert generate_house(43).snapshot() != a.snapshot()


def test_core_devices_exist():
    h = generate_house(1)
    for did in ["hallway.lock_front_door", "hallway.thermostat", "living_room.tv", "living_room.speaker", "bedroom.blinds"]:
        assert did in h.devices
    assert h.thermostat() is not None


def test_set_device_validates():
    h = generate_house(1)
    with pytest.raises(WorldError):
        h.set_device("nope.device", "power", "on")
    with pytest.raises(WorldError):
        h.set_device("living_room.tv", "brightness", 10)
    with pytest.raises(WorldError):
        h.set_device("living_room.light_main", "brightness", 250)
    with pytest.raises(WorldError):
        h.set_device("hallway.lock_front_door", "locked", "maybe")


def test_set_device_coerces_strings():
    h = generate_house(1)
    h.set_device("living_room.light_main", "brightness", "35")
    assert h.devices["living_room.light_main"].state["brightness"] == 35
    assert h.devices["living_room.light_main"].state["power"] == "on"
    h.set_device("hallway.lock_front_door", "locked", "TRUE")
    assert h.devices["hallway.lock_front_door"].state["locked"] is True
    h.set_device("living_room.blinds", "position", "Closed")
    assert h.devices["living_room.blinds"].state["position"] == "closed"


def test_light_state_is_canonical():
    h = generate_house(1)
    light = h.devices["living_room.light_main"]
    h.set_device(light.id, "brightness", 50)
    h.set_device(light.id, "power", "off")
    assert light.state == {"power": "off", "brightness": 0, "color_temp": light.state["color_temp"]}
    h.set_device(light.id, "power", "on")
    assert light.state["brightness"] == 100
    h.set_device(light.id, "brightness", 0)
    assert light.state["power"] == "off"


def test_schedule_and_lists():
    h = generate_house(2)
    a = h.add_schedule("hallway.thermostat", "mode", "heat", "Friday 17:00")
    assert a.when == "friday 17:00"
    with pytest.raises(WorldError):
        h.add_schedule("hallway.thermostat", "mode", "heat", "someday")
    h.add_to_list("shopping", "Milk")
    h.add_to_list("shopping", "milk")  # de-duplicated case-insensitively
    assert h.lists["shopping"].count("Milk") == 1
    with pytest.raises(WorldError):
        h.add_to_list("wishlist", "pony")


def test_snapshot_room_filter():
    h = generate_house(2)
    snap = h.snapshot("kitchen")
    assert list(snap["rooms"]) == ["kitchen"]
    with pytest.raises(WorldError):
        h.snapshot("attic")

"""House model and deterministic house generator.

The house is the "terrain" of the original game idea: rooms are locations,
devices are the objects the assistant can act on. Everything is generated
from a seed so training data, held-out evaluation, and before/after model
comparisons are reproducible.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from typing import Any

ROOM_LABELS = {
    "living_room": "living room",
    "kitchen": "kitchen",
    "bedroom": "bedroom",
    "guest_room": "guest room",
    "office": "office",
    "hallway": "hallway",
    "kids_room": "kids' room",
    "bathroom": "bathroom",
}

CORE_ROOMS = ["living_room", "kitchen", "bedroom", "hallway"]
OPTIONAL_ROOMS = ["guest_room", "office", "kids_room", "bathroom"]

WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]

# Attribute domains per device kind. Values are either a list of allowed
# strings/bools or a (min, max) integer range.
DEVICE_ATTRIBUTES: dict[str, dict[str, Any]] = {
    "light": {"power": ["on", "off"], "brightness": (0, 100), "color_temp": ["warm", "neutral", "cool"]},
    "thermostat": {"mode": ["heat", "cool", "eco", "off"], "target": (10, 30)},
    "lock": {"locked": [True, False]},
    "blinds": {"position": ["open", "closed"]},
    "tv": {"power": ["on", "off"], "volume": (0, 100)},
    "speaker": {"power": ["on", "off"], "volume": (0, 100)},
    "plug": {"power": ["on", "off"]},
}


class WorldError(Exception):
    """Raised for invalid actions against the house. The message is returned to the model."""


@dataclass
class Device:
    id: str
    kind: str
    name: str
    room: str
    state: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "kind": self.kind, "name": self.name, "room": self.room, **self.state}


@dataclass
class Room:
    id: str
    label: str
    temperature: int
    occupants: list[str] = field(default_factory=list)


@dataclass
class ScheduledAction:
    device_id: str
    attribute: str
    value: Any
    when: str

    def to_dict(self) -> dict[str, Any]:
        return {"device_id": self.device_id, "attribute": self.attribute, "value": self.value, "when": self.when}


@dataclass
class House:
    seed: int
    rooms: dict[str, Room]
    devices: dict[str, Device]
    hour: int
    weekday: str
    outside_temp: int
    user_location: str | None
    lists: dict[str, list[str]] = field(default_factory=lambda: {"shopping": [], "todo": []})
    schedule: list[ScheduledAction] = field(default_factory=list)

    # ---- derived properties -------------------------------------------------
    @property
    def is_dark(self) -> bool:
        return self.hour < 7 or self.hour >= 19

    @property
    def is_night(self) -> bool:
        """Night-time security window used by house rules."""
        return self.hour >= 22 or self.hour < 6

    def devices_in(self, room: str) -> list[Device]:
        return [d for d in self.devices.values() if d.room == room]

    def devices_of_kind(self, kind: str, room: str | None = None) -> list[Device]:
        return [d for d in self.devices.values() if d.kind == kind and (room is None or d.room == room)]

    def thermostat(self) -> Device | None:
        ts = self.devices_of_kind("thermostat")
        return ts[0] if ts else None

    def occupied_rooms(self) -> set[str]:
        return {r.id for r in self.rooms.values() if r.occupants}

    # ---- mutation -----------------------------------------------------------
    def set_device(self, device_id: str, attribute: str, value: Any) -> Device:
        device = self.devices.get(device_id)
        if device is None:
            raise WorldError(f"unknown device_id '{device_id}'. Known devices: {', '.join(sorted(self.devices))}")
        domain = DEVICE_ATTRIBUTES[device.kind]
        if attribute not in domain:
            raise WorldError(
                f"device '{device_id}' ({device.kind}) has no attribute '{attribute}'. Valid: {', '.join(domain)}"
            )
        value = coerce_value(domain[attribute], value, attribute)
        device.state[attribute] = value
        # Keep light state canonical: off <=> brightness 0, so different action
        # sequences that reach the same physical state compare equal.
        if device.kind == "light":
            if attribute == "brightness":
                device.state["power"] = "on" if value > 0 else "off"
            elif attribute == "power":
                if value == "off":
                    device.state["brightness"] = 0
                elif device.state["brightness"] == 0:
                    device.state["brightness"] = 100
        elif device.kind in ("tv", "speaker"):
            if attribute == "volume":
                device.state["power"] = "on" if value > 0 else "off"
            elif attribute == "power":
                if value == "off":
                    device.state["volume"] = 0
                elif device.state["volume"] == 0:
                    device.state["volume"] = 20
        return device

    def add_schedule(self, device_id: str, attribute: str, value: Any, when: str) -> ScheduledAction:
        device = self.devices.get(device_id)
        if device is None:
            raise WorldError(f"unknown device_id '{device_id}'")
        domain = DEVICE_ATTRIBUTES[device.kind]
        if attribute not in domain:
            raise WorldError(f"device '{device_id}' ({device.kind}) has no attribute '{attribute}'")
        value = coerce_value(domain[attribute], value, attribute)
        when = normalize_when(when)
        action = ScheduledAction(device_id, attribute, value, when)
        self.schedule.append(action)
        return action

    def add_to_list(self, list_name: str, item: str) -> list[str]:
        if list_name not in self.lists:
            raise WorldError(f"unknown list '{list_name}'. Valid lists: {', '.join(self.lists)}")
        item = " ".join(str(item).strip().split())
        if not item:
            raise WorldError("item must not be empty")
        if item.lower() not in {x.lower() for x in self.lists[list_name]}:
            self.lists[list_name].append(item)
        return list(self.lists[list_name])

    # ---- serialisation ------------------------------------------------------
    def snapshot(self, room: str | None = None) -> dict[str, Any]:
        """Compact JSON view. This is what goes into the model's context."""
        if room is not None and room not in self.rooms:
            raise WorldError(f"unknown room '{room}'. Rooms: {', '.join(self.rooms)}")
        rooms = [self.rooms[room]] if room else list(self.rooms.values())
        out: dict[str, Any] = {
            "time": f"{self.weekday} {self.hour:02d}:00",
            "outside": {"temperature": self.outside_temp, "dark": self.is_dark},
            "user_location": self.user_location,
            "rooms": {
                r.id: {
                    "label": r.label,
                    "temperature": r.temperature,
                    "occupants": list(r.occupants),
                    "devices": [d.to_dict() for d in self.devices_in(r.id)],
                }
                for r in rooms
            },
        }
        if room is None:
            out["lists"] = {k: list(v) for k, v in self.lists.items()}
            out["schedule"] = [s.to_dict() for s in self.schedule]
        return out

    def device_states(self) -> dict[str, dict[str, Any]]:
        return {d.id: dict(d.state) for d in self.devices.values()}

    def clone(self) -> "House":
        return copy.deepcopy(self)


# ---- helpers ------------------------------------------------------------------

def coerce_value(domain: Any, value: Any, attribute: str) -> Any:
    """Validate a value against an attribute domain, coercing strings from the model."""
    if isinstance(domain, tuple):
        try:
            ivalue = int(value)
        except (TypeError, ValueError):
            raise WorldError(f"attribute '{attribute}' expects an integer in {domain[0]}..{domain[1]}, got {value!r}")
        if not domain[0] <= ivalue <= domain[1]:
            raise WorldError(f"attribute '{attribute}' must be in {domain[0]}..{domain[1]}, got {ivalue}")
        return ivalue
    if domain == [True, False]:
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.lower() in {"true", "false"}:
            return value.lower() == "true"
        raise WorldError(f"attribute '{attribute}' expects true or false, got {value!r}")
    if isinstance(value, str) and value.lower() in domain:
        return value.lower()
    raise WorldError(f"attribute '{attribute}' must be one of {', '.join(map(str, domain))}, got {value!r}")


def normalize_when(when: Any) -> str:
    """Accept 'friday 17:00' style strings; normalise case and spacing."""
    if not isinstance(when, str):
        raise WorldError("when must be a string like 'friday 17:00'")
    parts = when.strip().lower().split()
    if len(parts) != 2 or parts[0] not in WEEKDAYS:
        raise WorldError("when must be '<weekday> HH:MM', e.g. 'friday 17:00'")
    hhmm = parts[1]
    if len(hhmm) != 5 or hhmm[2] != ":" or not (hhmm[:2] + hhmm[3:]).isdigit():
        raise WorldError("when must be '<weekday> HH:MM', e.g. 'friday 17:00'")
    h, m = int(hhmm[:2]), int(hhmm[3:])
    if not (0 <= h < 24 and 0 <= m < 60):
        raise WorldError("invalid time in when")
    return f"{parts[0]} {h:02d}:{m:02d}"


# ---- generator ----------------------------------------------------------------

def _light(room: str, name: str, rng: random.Random, label: str) -> Device:
    power = rng.choice(["on", "off"])
    return Device(
        id=f"{room}.{name}",
        kind="light",
        name=label,
        room=room,
        state={
            "power": power,
            "brightness": rng.choice([40, 60, 80, 100]) if power == "on" else 0,
            "color_temp": rng.choice(["warm", "neutral", "cool"]),
        },
    )


def generate_house(seed: int) -> House:
    """Build a plausible, fully random-but-seeded house."""
    rng = random.Random(seed)
    room_ids = list(CORE_ROOMS) + rng.sample(OPTIONAL_ROOMS, k=rng.randint(1, 3))

    hour = rng.randint(0, 23)
    weekday = rng.choice(WEEKDAYS)
    outside_temp = rng.randint(-2, 32)

    devices: dict[str, Device] = {}

    def add(d: Device) -> None:
        devices[d.id] = d

    for room in room_ids:
        add(_light(room, "light_main", rng, "main light"))

    # living room extras
    add(_light("living_room", "light_lamp", rng, "floor lamp"))
    add(Device("living_room.blinds", "blinds", "blinds", "living_room", {"position": rng.choice(["open", "closed"])}))
    tv_power = rng.choice(["on", "off"])
    add(Device("living_room.tv", "tv", "TV", "living_room", {"power": tv_power, "volume": rng.choice([10, 20, 30, 40]) if tv_power == "on" else 0}))
    sp_power = rng.choice(["on", "off"])
    add(Device("living_room.speaker", "speaker", "speaker", "living_room", {"power": sp_power, "volume": rng.choice([10, 20, 30]) if sp_power == "on" else 0}))

    # bedroom extras
    add(Device("bedroom.blinds", "blinds", "blinds", "bedroom", {"position": rng.choice(["open", "closed"])}))
    if rng.random() < 0.6:
        add(_light("bedroom", "light_lamp", rng, "bedside lamp"))

    # kitchen extras
    appliance = rng.choice(["kettle", "coffee_machine"])
    add(Device(f"kitchen.plug_{appliance}", "plug", appliance.replace("_", " "), "kitchen", {"power": "off"}))
    if rng.random() < 0.5:
        add(Device("kitchen.lock_back_door", "lock", "back door", "kitchen", {"locked": rng.choice([True, False])}))

    # hallway: front door + thermostat
    add(Device("hallway.lock_front_door", "lock", "front door", "hallway", {"locked": rng.choice([True, False])}))
    mode = rng.choice(["heat", "eco", "off", "cool"])
    add(Device("hallway.thermostat", "thermostat", "thermostat", "hallway", {"mode": mode, "target": rng.randint(15, 24)}))

    if "guest_room" in room_ids:
        add(Device("guest_room.plug_heater", "plug", "portable heater", "guest_room", {"power": "off"}))
    if "office" in room_ids:
        add(Device("office.blinds", "blinds", "blinds", "office", {"position": rng.choice(["open", "closed"])}))
    if "kids_room" in room_ids:
        add(Device("kids_room.speaker", "speaker", "speaker", "kids_room", {"power": "off", "volume": 0}))

    # occupants
    rooms: dict[str, Room] = {}
    base_temp = devices["hallway.thermostat"].state["target"]
    for rid in room_ids:
        rooms[rid] = Room(id=rid, label=ROOM_LABELS[rid], temperature=base_temp + rng.randint(-2, 2))

    user_location: str | None = rng.choice(room_ids + [None])
    if user_location:
        rooms[user_location].occupants.append("you")
    if "kids_room" in room_ids and rng.random() < 0.7:
        rooms["kids_room"].occupants.append("kids")
    if rng.random() < 0.4:
        partner_room = rng.choice(room_ids)
        rooms[partner_room].occupants.append("partner")

    house = House(
        seed=seed,
        rooms=rooms,
        devices=devices,
        hour=hour,
        weekday=weekday,
        outside_temp=outside_temp,
        user_location=user_location,
    )
    # a few pre-existing list items so add_to_list has realistic context
    if rng.random() < 0.5:
        house.lists["shopping"].append(rng.choice(["bread", "apples", "washing up liquid"]))
    if rng.random() < 0.3:
        house.lists["todo"].append(rng.choice(["call the plumber", "return library books"]))
    return house

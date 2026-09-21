"""Rule-coded oracle: given a scenario spec and the house, produce the ideal trajectory.

This is the "agent with a few simple rules" from the original plan. It never
sees the English utterance; it acts on the structured spec. Its trajectories
become training data, and its final house state is the evaluation target.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from homesim.intents import Scenario
from homesim.knowledge import HouseProfile, generate_profile
from homesim.world import House, ROOM_LABELS


@dataclass
class Step:
    note: str
    name: str
    arguments: dict[str, Any]


@dataclass
class Plan:
    steps: list[Step]
    expected_terminal: str  # "done" | "ask_user"

    @property
    def state_calls(self) -> list[Step]:
        return [s for s in self.steps if s.name in ("set_device", "create_schedule", "add_to_list")]


class _Planner:
    def __init__(self, house: House, profile: HouseProfile):
        self.h = house.clone()  # progressive state so no-ops are skipped correctly
        self.p = profile
        self.steps: list[Step] = []

    # -- primitives ---------------------------------------------------------
    def search(self, query: str, note: str) -> None:
        self.steps.append(Step(note, "search_context", {"query": query}))

    def want(self, device_id: str, attribute: str, value: Any, note: str | None = None) -> bool:
        d = self.h.devices[device_id]
        if d.state.get(attribute) == value:
            return False
        # Turning a light off also zeroes brightness etc.; apply to the clone so later checks are right.
        self.h.set_device(device_id, attribute, value)
        note = note or f"Set {d.name} in the {ROOM_LABELS[d.room]}: {attribute} -> {value}."
        self.steps.append(Step(note, "set_device", {"device_id": device_id, "attribute": attribute, "value": value}))
        return True

    def schedule(self, device_id: str, attribute: str, value: Any, when: str, note: str) -> None:
        self.h.add_schedule(device_id, attribute, value, when)
        self.steps.append(Step(note, "create_schedule", {"device_id": device_id, "attribute": attribute, "value": value, "when": when}))

    def add_list(self, list_name: str, item: str, note: str) -> bool:
        if item.lower() in {x.lower() for x in self.h.lists[list_name]}:
            return False
        self.h.add_to_list(list_name, item)
        self.steps.append(Step(note, "add_to_list", {"list_name": list_name, "item": item}))
        return True

    def done(self, message: str) -> Plan:
        self.steps.append(Step("Finished.", "done", {"message": message}))
        return Plan(self.steps, "done")

    def ask(self, question: str, note: str) -> Plan:
        self.steps.append(Step(note, "ask_user", {"question": question}))
        return Plan(self.steps, "ask_user")

    # -- group helpers --------------------------------------------------------
    def all_lights_off(self) -> None:
        for d in self.h.devices_of_kind("light"):
            self.want(d.id, "power", "off")

    def media_off(self, rooms: set[str] | None = None) -> None:
        for d in self.h.devices.values():
            if d.kind in ("tv", "speaker") and (rooms is None or d.room in rooms):
                self.want(d.id, "power", "off")

    def plugs_off(self) -> None:
        for d in self.h.devices_of_kind("plug"):
            self.want(d.id, "power", "off")

    def lock_all(self) -> None:
        for d in self.h.devices_of_kind("lock"):
            self.want(d.id, "locked", True)

    def thermostat(self, mode: str | None, target: int | None) -> None:
        t = self.h.thermostat()
        if t is None:
            return
        if mode is not None:
            self.want(t.id, "mode", mode)
        if target is not None:
            self.want(t.id, "target", target)


def plan(house: House, scenario: Scenario, profile: HouseProfile | None = None) -> Plan:
    profile = profile or generate_profile(house.seed)
    pl = _Planner(house, profile)
    p = profile
    h = pl.h
    name = scenario.intent
    params = scenario.params

    if scenario.ambiguous_room:
        return pl.ask("Which room do you mean?", "The room is not stated and I don't know where you are; asking.")

    if name == "movie_mode":
        pl.search("movie mode scene living room", "Looking up this household's movie scene.")
        for d in h.devices_of_kind("light", "living_room"):
            pl.want(d.id, "brightness", p.movie_brightness)
            pl.want(d.id, "color_temp", "warm")
        pl.want("living_room.blinds", "position", "closed")
        pl.want("living_room.tv", "power", "on")
        pl.want("living_room.speaker", "power", "on")
        pl.want("living_room.speaker", "volume", p.movie_volume)
        return pl.done("Movie mode is set in the living room. Enjoy the film!")

    if name == "bedtime":
        pl.search("bedtime routine", "Looking up the bedtime routine.")
        pl.all_lights_off()
        pl.media_off()
        pl.want("bedroom.blinds", "position", "closed")
        pl.lock_all()
        pl.thermostat("heat", p.night_temp)
        return pl.done("Good night. Lights off, doors locked, heating set for the night.")

    if name == "leaving_home":
        pl.search("leaving the house away mode", "Looking up the leaving-home routine.")
        pl.all_lights_off()
        pl.media_off()
        pl.plugs_off()
        pl.lock_all()
        pl.thermostat("eco", p.eco_temp)
        return pl.done("The house is in away mode: everything off, doors locked, heating on eco.")

    if name == "arriving_home":
        pl.search("arriving home routine", "Looking up the arriving-home routine.")
        pl.want("hallway.light_main", "brightness", p.hallway_welcome_brightness)
        pl.thermostat("heat", p.comfort_temp)
        if h.is_dark:
            pl.want("living_room.light_main", "brightness", p.reading_light)
        return pl.done("Welcome home! Hallway light on and heating set to comfort.")

    if name == "wake_up":
        pl.search("morning wake-up routine", "Looking up the morning routine.")
        pl.want("bedroom.blinds", "position", "open")
        pl.want("bedroom.light_main", "brightness", p.morning_brightness)
        pl.want("bedroom.light_main", "color_temp", "warm")
        plug = next(d for d in h.devices_of_kind("plug", "kitchen"))
        pl.want(plug.id, "power", "on")
        pl.thermostat("heat", p.comfort_temp)
        return pl.done("Good morning! Blinds open, bedroom light on, kitchen going and heating set to comfort.")

    if name == "too_cold":
        pl.search("cold comfort temperature preference", "Checking the temperature preference.")
        t = h.thermostat()
        assert t is not None
        target = p.comfort_temp if t.state["target"] < p.comfort_temp else min(30, t.state["target"] + 2)
        pl.thermostat("heat", target)
        return pl.done(f"Heating set to {target} degrees. It should warm up shortly.")

    if name == "too_hot":
        pl.search("too hot temperature preference", "Checking the temperature preference.")
        t = h.thermostat()
        assert t is not None
        target = max(15, t.state["target"] - 2)
        pl.thermostat(None, target)
        return pl.done(f"Thermostat lowered to {target} degrees.")

    if name == "too_bright":
        room = scenario.room
        assert room
        pl.search(f"too bright lighting preference {ROOM_LABELS[room]}", "Checking the lighting preference.")
        for d in h.devices_of_kind("light", room):
            if d.state["power"] == "on":
                pl.want(d.id, "brightness", p.low_light)
        if not h.is_dark:
            for b in h.devices_of_kind("blinds", room):
                pl.want(b.id, "position", "closed")
        return pl.done(f"Dimmed the {ROOM_LABELS[room]}.")

    if name == "too_dark":
        room = scenario.room
        assert room
        pl.search(f"too dark lighting preference {ROOM_LABELS[room]}", "Checking the lighting preference and rules.")
        level = p.reading_light
        if room == "kids_room" and h.hour >= 20:
            level = min(level, p.kids_light_cap)
        for d in h.devices_of_kind("light", room):
            pl.want(d.id, "brightness", level)
        if not h.is_dark:
            for b in h.devices_of_kind("blinds", room):
                pl.want(b.id, "position", "open")
        return pl.done(f"Brightened the {ROOM_LABELS[room]}.")

    if name == "guest_visit":
        pl.search("guests coming to stay guest room", "Checking the guest preparation preference.")
        when = f"{params['day']} {p.guest_warmup_time}"
        pl.schedule("guest_room.plug_heater", "power", "on", when, f"Scheduling the guest room heater for {when}.")
        pl.add_list("todo", "make up the guest bed", "Adding the guest bed to the todo list.")
        return pl.done(f"Guest room heater scheduled for {when} and 'make up the guest bed' is on your todo list.")

    if name == "shopping_item":
        if pl.add_list("shopping", params["item"], f"Adding {params['item']} to the shopping list."):
            return pl.done(f"Added {params['item']} to the shopping list.")
        return pl.done(f"{params['item']} is already on the shopping list.")

    if name == "todo_item":
        if pl.add_list("todo", params["item"], "Adding it to the todo list."):
            return pl.done(f"Added '{params['item']}' to your todo list.")
        return pl.done(f"'{params['item']}' is already on your todo list.")

    if name == "lock_up":
        pl.lock_all()
        return pl.done("All doors are locked.")

    if name == "unlock_door":
        pl.search("unlock door rule", "Checking the door rules before unlocking.")
        door_id, door = params["door_id"], params["door"]
        if h.is_night:
            return pl.ask(
                f"It's after 22:00, so I can't unlock the {door} from chat. Please confirm the unlock in the phone app.",
                "Night-time rule: unlocking needs confirmation; asking.",
            )
        pl.want(door_id, "locked", False)
        return pl.done(f"The {door} is unlocked.")

    if name == "status_query":
        return pl.done(_status_answer(h, params["subject"]))

    if name == "no_action":
        return pl.done("You're welcome! Let me know if you need anything else.")

    if name == "energy_save":
        pl.search("energy saving mode", "Checking the energy saving preference.")
        empty = {r for r in h.rooms if not h.rooms[r].occupants}
        for d in list(h.devices.values()):
            if d.room in empty and d.kind == "light":
                pl.want(d.id, "power", "off")
        pl.media_off(empty)
        pl.thermostat("eco", None)
        return pl.done("Energy saving on: unoccupied rooms powered down and heating on eco.")

    raise ValueError(f"unknown intent {name}")


def _status_answer(h: House, subject: str) -> str:
    if subject == "front_door":
        return "The front door is locked." if h.devices["hallway.lock_front_door"].state["locked"] else "The front door is unlocked."
    if subject == "back_door":
        return "The back door is locked." if h.devices["kitchen.lock_back_door"].state["locked"] else "The back door is unlocked."
    if subject == "thermostat":
        t = h.thermostat()
        assert t is not None
        return f"The thermostat is in {t.state['mode']} mode with a target of {t.state['target']} degrees."
    if subject == "shopping":
        items = h.lists["shopping"]
        return "The shopping list has: " + ", ".join(items) + "." if items else "The shopping list is empty."
    if subject == "tv":
        return "The TV is on." if h.devices["living_room.tv"].state["power"] == "on" else "The TV is off."
    if subject.startswith("lights:"):
        room = subject.split(":", 1)[1]
        on = [d.name for d in h.devices_of_kind("light", room) if d.state["power"] == "on"]
        return f"In the {ROOM_LABELS[room]} these lights are on: {', '.join(on)}." if on else f"No lights are on in the {ROOM_LABELS[room]}."
    raise ValueError(subject)


@dataclass
class OraclePolicy:
    """Replays a plan one tool call per assistant turn, or all state calls in one turn when batch=True."""

    plan: Plan
    batch: bool = False
    _cursor: int = field(default=0, repr=False)

    def __call__(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
        from homesim.episode import assistant_message

        steps = self.plan.steps
        if self._cursor >= len(steps):
            return assistant_message("Finished.", [("done", {"message": "Done."})])
        if not self.batch:
            s = steps[self._cursor]
            self._cursor += 1
            return assistant_message(s.note, [(s.name, s.arguments)])
        # batch mode: search alone, then every state call together, then the terminal call alone
        s = steps[self._cursor]
        if s.name in ("search_context", "done", "ask_user"):
            self._cursor += 1
            return assistant_message(s.note, [(s.name, s.arguments)])
        group: list[Step] = []
        while self._cursor < len(steps) and steps[self._cursor].name not in ("search_context", "done", "ask_user"):
            group.append(steps[self._cursor])
            self._cursor += 1
        return assistant_message(" ".join(g.note for g in group), [(g.name, g.arguments) for g in group])

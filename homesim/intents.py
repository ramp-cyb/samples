"""Intent catalogue: structured scenario specs and the English templates that express them.

Every intent has a fixed, rule-executable meaning (see oracle.py). Templates
deliberately avoid the tool vocabulary and often rely on world knowledge
("the milk's gone off" -> shopping list; "we're about to start a film" -> movie
scene). Template ids allow holding out whole phrasings for evaluation.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable

from homesim.world import House, ROOM_LABELS, WEEKDAYS

# ---- slot vocabularies ----------------------------------------------------------

SHOPPING_ITEMS = [
    "milk", "eggs", "bread", "coffee", "bin bags", "washing powder", "toothpaste", "olive oil",
    "bananas", "toilet roll", "dish soap", "butter", "rice", "cat food", "tomatoes", "cheese",
]
PERISHABLES = {"milk", "eggs", "bread", "bananas", "butter", "tomatoes", "cheese"}
TODO_ITEMS = [
    "call the dentist", "book the car service", "pay the electricity bill", "fix the garden gate",
    "renew the parking permit", "email the landlord about the tap", "order a new filter for the hoover",
    "book a table for Saturday", "return the parcel", "sort out the recycling",
]
DAY_PHRASES = {
    "monday": "on Monday", "tuesday": "on Tuesday", "wednesday": "on Wednesday", "thursday": "on Thursday",
    "friday": "on Friday", "saturday": "this weekend", "sunday": "on Sunday",
}


@dataclass
class Template:
    text: str
    needs_room: bool = False  # template contains {room}; if False, room comes from user location


@dataclass
class IntentDef:
    name: str
    templates: list[Template]
    room_scoped: bool = False  # oracle needs a resolved room
    applies: Callable[[House], bool] = lambda h: True
    sample_params: Callable[[House, random.Random], dict[str, Any]] = lambda h, r: {}


@dataclass
class Scenario:
    intent: str
    params: dict[str, Any]
    template_id: str
    utterance: str
    room: str | None = None  # resolved room for room-scoped intents
    ambiguous_room: bool = False  # room needed but unresolvable -> ask_user expected
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent": self.intent,
            "params": self.params,
            "template_id": self.template_id,
            "utterance": self.utterance,
            "room": self.room,
            "ambiguous_room": self.ambiguous_room,
        }


def _t(*texts: str, needs_room: bool = False) -> list[Template]:
    return [Template(t, needs_room) for t in texts]


def _rooms_with_lights(h: House) -> list[str]:
    return [r for r in h.rooms if h.devices_of_kind("light", r)]


def _pick_room(h: House, rng: random.Random) -> dict[str, Any]:
    return {"room": rng.choice(_rooms_with_lights(h))}


def _pick_shopping(h: House, rng: random.Random) -> dict[str, Any]:
    return {"item": rng.choice(SHOPPING_ITEMS)}


def _pick_todo(h: House, rng: random.Random) -> dict[str, Any]:
    return {"item": rng.choice(TODO_ITEMS)}


def _pick_day(h: House, rng: random.Random) -> dict[str, Any]:
    day = rng.choice(WEEKDAYS)
    return {"day": day, "day_phrase": DAY_PHRASES[day]}


def _pick_door(h: House, rng: random.Random) -> dict[str, Any]:
    locks = h.devices_of_kind("lock")
    d = rng.choice(locks)
    return {"door_id": d.id, "door": d.name}


def _pick_status(h: House, rng: random.Random) -> dict[str, Any]:
    subjects = ["front_door", "thermostat", "shopping", "tv"] + [f"lights:{r}" for r in _rooms_with_lights(h)]
    if "kitchen.lock_back_door" in h.devices:
        subjects.append("back_door")
    return {"subject": rng.choice(subjects)}


INTENTS: dict[str, IntentDef] = {}


def _register(d: IntentDef) -> None:
    INTENTS[d.name] = d


_register(IntentDef(
    "movie_mode",
    _t(
        "We're about to start a film in the living room.",
        "Movie night! Set the living room up for it.",
        "Popcorn's ready, we're putting a movie on.",
        "Can you get the room ready for a film?",
        "Let's watch something, make it cosy in the living room.",
        "Film time. You know the drill.",
        "The kids want to watch a movie in the living room, sort the lights and stuff out please.",
        "Setting up for a movie, could you do the usual?",
        "Cinema mode please.",
        "Time for our Friday film, get the living room ready.",
    ),
))

_register(IntentDef(
    "bedtime",
    _t(
        "I'm heading to bed.",
        "Good night, house.",
        "We're turning in for the night.",
        "Time to sleep, shut everything down.",
        "Off to bed now, can you sort the house out?",
        "Night night. Lock up and lights out please.",
        "I'm done for today, going to sleep.",
        "Bedtime routine please.",
        "Calling it a night.",
        "Everyone's asleep already, I'm going up too.",
    ),
))

_register(IntentDef(
    "leaving_home",
    _t(
        "I'm off to the office, back tonight.",
        "We're all leaving now, see you later.",
        "Heading out for the day.",
        "Leaving the house, nobody's home until this evening.",
        "Off to work. Can you take care of things?",
        "We're going to grandma's for the weekend, house is empty.",
        "Just leaving, everything off please.",
        "Going out now, set the house to away.",
        "Everybody out, shutting the door behind me.",
        "Leaving for the airport, back in three days.",
    ),
))

_register(IntentDef(
    "arriving_home",
    _t(
        "I'm home!",
        "Just walked in the door.",
        "Back from work, finally.",
        "We're back.",
        "Home sweet home. Warm it up in here.",
        "I've just got in, it's freezing.",
        "Made it home, do the welcome thing.",
        "Just pulled into the driveway.",
        "Home again, long day.",
        "I'm back, make the place welcoming.",
    ),
))

_register(IntentDef(
    "wake_up",
    _t(
        "Good morning.",
        "I'm up, start the day.",
        "Morning! Rise and shine.",
        "Just woke up, do the morning routine.",
        "Ugh, morning. Get things going.",
        "Wake-up routine please.",
        "Time to get up. Coffee and light please.",
        "I'm awake, open things up.",
        "Morning, let some light in and get the kitchen going.",
        "Alarm just went off, start the morning stuff.",
    ),
))

_register(IntentDef(
    "too_cold",
    _t(
        "It's freezing in here.",
        "I'm cold.",
        "Brr, can you warm the place up a bit?",
        "It's a bit chilly, turn the heating up.",
        "My hands are like ice, it's too cold in the {room}.",
        "Could we have it warmer in the {room}?",
        "Bit nippy today, more heat please.",
        "The house feels cold, bump the temperature up.",
        "I'm shivering in the {room}.",
        "Can you make it warmer?",
    ),
    sample_params=_pick_room,
))

_register(IntentDef(
    "too_hot",
    _t(
        "It's boiling in here.",
        "I'm too hot, turn the heating down.",
        "It's stuffy and warm, cool it down a bit.",
        "Way too warm in the {room}.",
        "Can we have it a bit cooler?",
        "I'm sweating, lower the temperature.",
        "The {room} is like a sauna.",
        "Too hot in the house, take it down a notch.",
        "Drop the heating a little please.",
        "Phew, it's roasting. Cooler please.",
    ),
    sample_params=_pick_room,
))

_register(IntentDef(
    "too_bright",
    _t(
        "It's too bright in here.",
        "Ow, my eyes. Can you dim the lights?",
        "The {room} is way too bright.",
        "Bit glaring in the {room}, tone it down.",
        "Too much light in here, soften it.",
        "Can you make it less bright in the {room}?",
        "It's like an operating theatre in here, dim it.",
        "I've got a headache, less light please.",
        "The sun's glaring in and the lights are on full in the {room}.",
        "Dim things down a bit.",
    ),
    room_scoped=True,
    sample_params=_pick_room,
))
for _tpl in INTENTS["too_bright"].templates:
    _tpl.needs_room = "{room}" in _tpl.text

_register(IntentDef(
    "too_dark",
    _t(
        "I can't see a thing in here.",
        "It's too dark in the {room}.",
        "Can you brighten the {room} up?",
        "It's gloomy in here, more light please.",
        "I'm trying to read and it's way too dim.",
        "Lights up in the {room}, I'm working.",
        "Bit dark in here.",
        "Could we have more light in the {room}?",
        "It's like a cave in here, brighten it.",
        "Turn the lights up, I'm doing a jigsaw in the {room}.",
    ),
    room_scoped=True,
    sample_params=_pick_room,
))
for _tpl in INTENTS["too_dark"].templates:
    _tpl.needs_room = "{room}" in _tpl.text

_register(IntentDef(
    "guest_visit",
    _t(
        "My parents are coming to stay {day_phrase}.",
        "We've got friends staying over {day_phrase}.",
        "Grandma's visiting {day_phrase}, she'll be in the guest room.",
        "Guests arriving {day_phrase}, get the spare room ready.",
        "My sister is staying with us {day_phrase}.",
        "We're having people to stay {day_phrase}.",
        "The in-laws land {day_phrase} and they're staying a few nights.",
        "A friend from uni is crashing here {day_phrase}.",
        "Prepare for house guests {day_phrase}.",
        "Visitors {day_phrase}, the usual guest prep please.",
    ),
    applies=lambda h: "guest_room" in h.rooms,
    sample_params=_pick_day,
))

_register(IntentDef(
    "shopping_item",
    _t(
        "We're out of {item}.",
        "The {item} has gone off, we need more.",
        "Add {item} to the shopping list.",
        "Remind me to buy {item}.",
        "No more {item} left in the cupboard.",
        "Put {item} on the list for the shop.",
        "We need {item} next time someone goes to the supermarket.",
        "Running low on {item}.",
        "I just used the last of the {item}.",
        "Note that we need {item}.",
    ),
    sample_params=_pick_shopping,
))

_register(IntentDef(
    "todo_item",
    _t(
        "Remind me to {item}.",
        "I need to {item} at some point, note it down.",
        "Add '{item}' to my todo list.",
        "Don't let me forget to {item}.",
        "Put '{item}' on the list of things to do.",
        "Todo: {item}.",
        "I keep forgetting to {item}, make a note.",
        "Jot down that I have to {item}.",
        "Make sure I {item} this week.",
        "Note to self: {item}.",
    ),
    sample_params=_pick_todo,
))

_register(IntentDef(
    "lock_up",
    _t(
        "Lock all the doors.",
        "Make sure the house is locked up.",
        "Is everything locked? Lock it if not.",
        "Secure the doors please.",
        "Lock up.",
        "I heard something outside, lock the doors.",
        "Bolt the doors.",
        "Can you lock everything?",
        "Double check the doors are locked.",
        "Doors locked please, all of them.",
    ),
))

_register(IntentDef(
    "unlock_door",
    _t(
        "Unlock the {door}.",
        "Open up the {door}, my hands are full.",
        "Let me in through the {door}.",
        "The delivery guy is at the {door}, unlock it.",
        "Can you unlock the {door}?",
        "I forgot my keys, unlock the {door} please.",
        "Pop the {door} open for the neighbour.",
        "Unlock the {door} for me.",
        "{door} unlocked please.",
        "The dog walker is here, let them in via the {door}.",
    ),
    sample_params=_pick_door,
))

_register(IntentDef(
    "status_query",
    _t(
        "{question}",
    ),
    sample_params=_pick_status,
))

_register(IntentDef(
    "no_action",
    _t(
        "Thanks, that's all.",
        "Great, cheers.",
        "Never mind.",
        "Nothing for now, thank you.",
        "Perfect, you can stop there.",
        "Ok that's it.",
        "Thanks house!",
        "No that's everything.",
        "All good, thanks.",
        "Forget it, it's fine.",
    ),
))

_register(IntentDef(
    "energy_save",
    _t(
        "The energy bill is huge, save some power.",
        "Go into energy saving mode.",
        "Turn off whatever's on in the empty rooms.",
        "Save energy please.",
        "Cut the electricity use where nobody is.",
        "Eco mode for the house.",
        "We're wasting power, sort it out.",
        "Switch to energy saving.",
        "Power down the rooms nobody's in.",
        "Be frugal with the electricity.",
    ),
))

STATUS_QUESTIONS: dict[str, list[str]] = {
    "front_door": ["Is the front door locked?", "Did I lock the front door?", "Front door status?"],
    "back_door": ["Is the back door locked?", "Did anyone lock the back door?", "What's the back door doing?"],
    "thermostat": ["What's the heating set to?", "What temperature is the thermostat on?", "Heating status?"],
    "shopping": ["What's on the shopping list?", "Read me the shopping list.", "Anything on the shopping list?"],
    "tv": ["Is the TV on?", "Did someone leave the telly on?", "TV status?"],
    "lights": ["Are the lights on in the {room}?", "Any lights on in the {room}?", "Is the {room} lit?"],
}


def render_utterance(intent: IntentDef, template: Template, params: dict[str, Any], house: House) -> str:
    slots = dict(params)
    if "room" in params:
        slots["room"] = ROOM_LABELS[params["room"]]
    if intent.name == "status_query":
        subject = params["subject"]
        key = "lights" if subject.startswith("lights:") else subject
        q_index = params.get("q_index", 0)
        q = STATUS_QUESTIONS[key][q_index % len(STATUS_QUESTIONS[key])]
        if key == "lights":
            q = q.format(room=ROOM_LABELS[subject.split(":", 1)[1]])
        return q
    text = template.text.format(**slots)
    return text[0].upper() + text[1:]


def sample_scenario(house: House, rng: random.Random, intent_name: str | None = None,
                    template_index: int | None = None) -> Scenario:
    candidates = [d for d in INTENTS.values() if d.applies(house)]
    intent = INTENTS[intent_name] if intent_name else rng.choice(candidates)
    params = intent.sample_params(house, rng)
    if intent.name == "status_query":
        params["q_index"] = rng.randrange(3)
    ti = template_index if template_index is not None else rng.randrange(len(intent.templates))
    template = intent.templates[ti]
    utterance = render_utterance(intent, template, params, house)
    room: str | None = None
    ambiguous = False
    if intent.room_scoped:
        if template.needs_room:
            room = params["room"]
        elif house.user_location and house.devices_of_kind("light", house.user_location):
            room = house.user_location
        else:
            ambiguous = True
    template_id = f"{intent.name}:{ti}" if intent.name != "status_query" else f"status_query:{params['subject'].split(':')[0]}:{params['q_index']}"
    return Scenario(intent.name, params, template_id, utterance, room=room, ambiguous_room=ambiguous)

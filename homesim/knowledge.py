"""Per-house knowledge base and a small BM25 retriever.

Scene parameters (movie brightness, night temperature, ...) are randomised per
house and written into documents. A model therefore cannot memorise them from
training data; it has to retrieve and read them. This is what gives the RAG
step a real job.
"""

from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass, field
from typing import Any

from homesim.world import House, ROOM_LABELS

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@dataclass
class Document:
    id: str
    title: str
    text: str
    kind: str  # scene | preference | rule | note | distractor

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "title": self.title, "text": self.text}


@dataclass
class HouseProfile:
    """Structured version of the knowledge base; the oracle reads this, the model reads the docs."""

    movie_brightness: int
    movie_volume: int
    night_temp: int
    eco_temp: int
    comfort_temp: int
    low_light: int
    reading_light: int
    hallway_welcome_brightness: int
    morning_brightness: int
    guest_warmup_time: str  # HH:MM
    kids_light_cap: int
    night_unlock_requires_confirmation: bool = True


def generate_profile(seed: int) -> HouseProfile:
    rng = random.Random(seed * 7919 + 17)
    return HouseProfile(
        movie_brightness=rng.choice([10, 15, 20, 25, 30]),
        movie_volume=rng.choice([20, 25, 30, 35, 40]),
        night_temp=rng.choice([16, 17, 18, 19]),
        eco_temp=rng.choice([14, 15, 16, 17]),
        comfort_temp=rng.choice([20, 21, 22, 23]),
        low_light=rng.choice([20, 30, 40]),
        reading_light=rng.choice([70, 80, 90, 100]),
        hallway_welcome_brightness=rng.choice([60, 80, 100]),
        morning_brightness=rng.choice([40, 50, 60]),
        guest_warmup_time=rng.choice(["15:00", "16:00", "17:00"]),
        kids_light_cap=rng.choice([40, 50, 60]),
    )


def build_documents(house: House, profile: HouseProfile) -> list[Document]:
    p = profile
    docs: list[Document] = []

    def add(id_: str, title: str, text: str, kind: str) -> None:
        docs.append(Document(id_, title, " ".join(text.split()), kind))

    add(
        "scene_movie",
        "Movie mode scene",
        f"Movie mode (film night, watching something) in the living room: set the main light and the floor lamp to "
        f"{p.movie_brightness}% brightness with warm colour temperature, close the blinds, turn the TV on, and turn the "
        f"speaker on at volume {p.movie_volume}.",
        "scene",
    )
    add(
        "scene_bedtime",
        "Bedtime routine",
        f"Bedtime (going to sleep, good night): turn every light off, turn the TV and all speakers off, close the bedroom "
        f"blinds, lock every door, and set the thermostat to heat mode with a target of {p.night_temp} degrees.",
        "scene",
    )
    add(
        "scene_away",
        "Leaving the house routine",
        f"When everyone leaves the house (going out, off to work, away mode): turn every light off, turn the TV, "
        f"speakers and smart plugs off, lock every door, and set the thermostat to eco mode with a target of "
        f"{p.eco_temp} degrees.",
        "scene",
    )
    add(
        "scene_welcome",
        "Arriving home routine",
        f"When someone arrives home (back from work, walking in the door): turn the hallway main light on at "
        f"{p.hallway_welcome_brightness}% brightness, set the thermostat to heat mode at the comfort temperature of "
        f"{p.comfort_temp} degrees, and if it is dark outside also turn the living room main light on at "
        f"{p.reading_light}%.",
        "scene",
    )
    add(
        "scene_morning",
        "Morning wake-up routine",
        f"Wake-up routine (good morning, getting up): open the bedroom blinds, turn the bedroom main light on at "
        f"{p.morning_brightness}% warm, turn the kitchen appliance plug (kettle or coffee machine) on, and set the "
        f"thermostat to heat mode at the comfort temperature of {p.comfort_temp} degrees.",
        "scene",
    )
    add(
        "pref_temperature",
        "Temperature preferences",
        f"The household comfort temperature is {p.comfort_temp} degrees. If someone is cold, set the thermostat to heat "
        f"mode at the comfort temperature; if the target is already at or above the comfort temperature, raise the "
        f"target by 2 degrees instead. If someone is too hot, lower the thermostat target by 2 degrees (never below 15) "
        f"and keep the mode unchanged.",
        "preference",
    )
    add(
        "pref_lighting",
        "Lighting preferences",
        f"When a room is too bright, dim every light that is on in that room to {p.low_light}% and close the room's blinds if it "
        f"is daytime. When a room is too dark, set every light in that room to {p.reading_light}% and open the room's "
        f"blinds if it is daytime and the room has blinds. Lights that are already at the requested level are left alone.",
        "preference",
    )
    add(
        "pref_guests",
        "Guest preparation",
        f"When guests are coming to stay: schedule the guest room portable heater to switch on at "
        f"{p.guest_warmup_time} on the day they arrive, and add 'make up the guest bed' to the todo list.",
        "preference",
    )
    add(
        "rule_night_lock",
        "House rule: doors at night",
        "Security rule: between 22:00 and 06:00 a door must never be unlocked from a voice or chat request. "
        "Ask the user to confirm the unlock on their phone app instead of unlocking.",
        "rule",
    )
    add(
        "rule_kids_light",
        "House rule: kids' room lighting",
        f"The kids' room lights must not be set above {p.kids_light_cap}% brightness after 20:00.",
        "rule",
    )
    add(
        "pref_energy",
        "Energy saving",
        "Energy saving mode: in every room with no occupants turn the lights, TV and speakers off, and set the "
        "thermostat to eco mode keeping its current target.",
        "preference",
    )
    add(
        "note_locks",
        "Door locks",
        "The front door lock is hallway.lock_front_door. "
        + ("The back door lock is kitchen.lock_back_door." if "kitchen.lock_back_door" in house.devices else "There is no smart lock on the back door."),
        "note",
    )
    appliance = next(d for d in house.devices.values() if d.kind == "plug" and d.room == "kitchen")
    add("note_kitchen", "Kitchen appliances", f"The {appliance.name} in the kitchen is on smart plug {appliance.id}.", "note")
    rooms = ", ".join(f"{ROOM_LABELS[r]} ({r})" for r in house.rooms)
    add("note_rooms", "Rooms in this house", f"This house has these rooms: {rooms}.", "note")

    # distractors: plausible household documents that should not be acted on
    add("misc_bins", "Bin collection", "General waste is collected on Tuesday mornings, recycling every other Thursday.", "distractor")
    add("misc_wifi", "Wi-Fi", "The guest Wi-Fi password is written on the card inside the kitchen drawer.", "distractor")
    add("misc_boiler", "Boiler service", "The boiler was last serviced in March; the engineer's number is on the fridge.", "distractor")
    add("misc_plants", "Plants", "Water the living room plants on Sundays; the orchid only needs water every two weeks.", "distractor")
    return docs


@dataclass
class KnowledgeBase:
    documents: list[Document]
    k1: float = 1.5
    b: float = 0.75
    _index: dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def for_house(cls, house: House, profile: HouseProfile | None = None) -> "KnowledgeBase":
        profile = profile or generate_profile(house.seed)
        return cls(build_documents(house, profile))

    def __post_init__(self) -> None:
        self._build()

    def _build(self) -> None:
        tokenized = [tokenize(d.title + " " + d.text) for d in self.documents]
        df: dict[str, int] = {}
        for toks in tokenized:
            for t in set(toks):
                df[t] = df.get(t, 0) + 1
        n = len(tokenized)
        avgdl = sum(len(t) for t in tokenized) / max(n, 1)
        self._index = {"tokens": tokenized, "df": df, "n": n, "avgdl": avgdl}

    def score(self, query: str, doc_idx: int) -> float:
        idx = self._index
        toks = idx["tokens"][doc_idx]
        if not toks:
            return 0.0
        tf: dict[str, int] = {}
        for t in toks:
            tf[t] = tf.get(t, 0) + 1
        score = 0.0
        dl = len(toks)
        for q in tokenize(query):
            if q not in tf:
                continue
            n_q = idx["df"][q]
            idf = math.log(1 + (idx["n"] - n_q + 0.5) / (n_q + 0.5))
            f = tf[q]
            score += idf * (f * (self.k1 + 1)) / (f + self.k1 * (1 - self.b + self.b * dl / idx["avgdl"]))
        return score

    def search(self, query: str, top_k: int = 3) -> list[Document]:
        scored = [(self.score(query, i), i) for i in range(len(self.documents))]
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [self.documents[i] for s, i in scored[:top_k] if s > 0]

    def get(self, doc_id: str) -> Document:
        for d in self.documents:
            if d.id == doc_id:
                return d
        raise KeyError(doc_id)

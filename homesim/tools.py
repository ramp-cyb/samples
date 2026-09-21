"""Tool schemas exposed to the model and the executor that runs them against a House.

The schema format is the OpenAI-style function schema that Qwen/Llama chat
templates accept in their `tools` argument.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from homesim.knowledge import KnowledgeBase
from homesim.world import House, WorldError

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_context",
            "description": (
                "Search the household knowledge base for scenes, preferences, house rules and device notes. "
                "Use this before applying a named routine (movie, bedtime, leaving, arriving, morning) or "
                "when a preference such as a comfort temperature or light level is needed."
            ),
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Short search query"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_state",
            "description": "Return the current state of the whole house, or of one room.",
            "parameters": {
                "type": "object",
                "properties": {"room": {"type": "string", "description": "Room id, e.g. living_room. Omit for the whole house."}},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_device",
            "description": (
                "Set one attribute of one device. Lights: power(on/off), brightness(0-100), color_temp(warm/neutral/cool). "
                "Thermostat: mode(heat/cool/eco/off), target(10-30). Lock: locked(true/false). Blinds: position(open/closed). "
                "TV and speaker: power(on/off), volume(0-100). Plug: power(on/off)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "device_id": {"type": "string", "description": "Device id, e.g. living_room.light_main"},
                    "attribute": {"type": "string"},
                    "value": {"type": ["string", "integer", "boolean"]},
                },
                "required": ["device_id", "attribute", "value"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_schedule",
            "description": "Schedule a device attribute change for a later time.",
            "parameters": {
                "type": "object",
                "properties": {
                    "device_id": {"type": "string"},
                    "attribute": {"type": "string"},
                    "value": {"type": ["string", "integer", "boolean"]},
                    "when": {"type": "string", "description": "'<weekday> HH:MM', e.g. 'friday 17:00'"},
                },
                "required": ["device_id", "attribute", "value", "when"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_to_list",
            "description": "Add an item to the shopping list or the todo list.",
            "parameters": {
                "type": "object",
                "properties": {
                    "list_name": {"type": "string", "enum": ["shopping", "todo"]},
                    "item": {"type": "string"},
                },
                "required": ["list_name", "item"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ask_user",
            "description": (
                "Ask the user a clarifying question and stop. Use when the request is ambiguous "
                "(for example the room is unknown) or when a house rule requires explicit confirmation."
            ),
            "parameters": {
                "type": "object",
                "properties": {"question": {"type": "string"}},
                "required": ["question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "done",
            "description": "Finish the request with a short confirmation or answer for the user.",
            "parameters": {
                "type": "object",
                "properties": {"message": {"type": "string"}},
                "required": ["message"],
            },
        },
    },
]

TOOL_NAMES = {t["function"]["name"] for t in TOOL_SCHEMAS}
TERMINAL_TOOLS = {"ask_user", "done"}


@dataclass
class ToolResult:
    name: str
    arguments: dict[str, Any]
    ok: bool
    result: Any
    terminal: bool = False
    changed: bool = False  # did the call change house state?

    def as_content(self) -> str:
        payload = {"ok": self.ok, "result": self.result}
        return json.dumps(payload, ensure_ascii=False)


@dataclass
class ToolExecutor:
    house: House
    kb: KnowledgeBase
    top_k: int = 3
    log: list[ToolResult] = field(default_factory=list)

    def execute(self, name: str, arguments: dict[str, Any] | None) -> ToolResult:
        arguments = arguments or {}
        try:
            res = self._dispatch(name, arguments)
        except WorldError as e:
            res = ToolResult(name, arguments, ok=False, result=f"error: {e}")
        except (TypeError, KeyError) as e:
            res = ToolResult(name, arguments, ok=False, result=f"error: bad arguments for {name}: {e}")
        self.log.append(res)
        return res

    def _require(self, arguments: dict[str, Any], *keys: str) -> None:
        missing = [k for k in keys if k not in arguments]
        if missing:
            raise WorldError(f"missing required argument(s): {', '.join(missing)}")

    def _dispatch(self, name: str, a: dict[str, Any]) -> ToolResult:
        if name not in TOOL_NAMES:
            raise WorldError(f"unknown tool '{name}'. Available: {', '.join(sorted(TOOL_NAMES))}")
        if name == "search_context":
            self._require(a, "query")
            docs = self.kb.search(str(a["query"]), top_k=self.top_k)
            return ToolResult(name, a, True, [d.to_dict() for d in docs])
        if name == "get_state":
            return ToolResult(name, a, True, self.house.snapshot(a.get("room")))
        if name == "set_device":
            self._require(a, "device_id", "attribute", "value")
            before = dict(self.house.devices[a["device_id"]].state) if a["device_id"] in self.house.devices else None
            device = self.house.set_device(a["device_id"], a["attribute"], a["value"])
            return ToolResult(name, a, True, device.to_dict(), changed=before != device.state)
        if name == "create_schedule":
            self._require(a, "device_id", "attribute", "value", "when")
            action = self.house.add_schedule(a["device_id"], a["attribute"], a["value"], a["when"])
            return ToolResult(name, a, True, action.to_dict(), changed=True)
        if name == "add_to_list":
            self._require(a, "list_name", "item")
            before = list(self.house.lists.get(a["list_name"], []))
            items = self.house.add_to_list(a["list_name"], a["item"])
            return ToolResult(name, a, True, {"list_name": a["list_name"], "items": items}, changed=before != items)
        if name == "ask_user":
            self._require(a, "question")
            return ToolResult(name, a, True, "waiting for the user's answer", terminal=True)
        if name == "done":
            self._require(a, "message")
            return ToolResult(name, a, True, "ok", terminal=True)
        raise WorldError(f"unhandled tool {name}")

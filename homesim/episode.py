"""Episode runner: the loop between a policy (oracle or LLM) and the house."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from homesim.knowledge import KnowledgeBase
from homesim.tools import TOOL_SCHEMAS, ToolExecutor, ToolResult
from homesim.world import House

SYSTEM_PROMPT = """You are the assistant for a smart home. You act only through tool calls.

Rules:
- The full current state of the house is given below. Read it before acting; do not change what is already in the requested state.
- Named routines (movie, bedtime, leaving, arriving, morning) and preferences (temperatures, light levels) differ per household. Call search_context first and follow the retrieved document exactly.
- If the room is not stated and cannot be inferred from where the user is, call ask_user.
- House rules from search_context take priority over the request. If a rule requires confirmation, call ask_user instead of acting.
- Finish every request with exactly one done or ask_user call.

House state:
{state}"""


class Policy(Protocol):
    """A policy receives the full message list and returns the next assistant message."""

    def __call__(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]: ...


@dataclass
class Trajectory:
    messages: list[dict[str, Any]]
    tool_results: list[ToolResult]
    terminal: str | None  # "done" | "ask_user" | None (ran out of turns)
    final_house: House
    turns: int
    parse_failures: int = 0
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def calls(self) -> list[dict[str, Any]]:
        return [{"name": r.name, "arguments": r.arguments} for r in self.tool_results]


def assistant_message(content: str, tool_calls: list[tuple[str, dict[str, Any]]]) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": content,
        "tool_calls": [
            {"type": "function", "function": {"name": name, "arguments": args}} for name, args in tool_calls
        ],
    }


def build_system_prompt(house: House) -> str:
    return SYSTEM_PROMPT.format(state=json.dumps(house.snapshot(), ensure_ascii=False))


def run_episode(
    house: House,
    kb: KnowledgeBase,
    policy: Policy,
    utterance: str,
    max_turns: int = 12,
    max_calls_per_turn: int = 8,
    on_turn: Callable[[dict[str, Any], list[ToolResult]], None] | None = None,
) -> Trajectory:
    """Run one request to completion. The house is mutated in place."""
    executor = ToolExecutor(house, kb)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": build_system_prompt(house)},
        {"role": "user", "content": utterance},
    ]
    terminal: str | None = None
    parse_failures = 0
    turns = 0
    while turns < max_turns and terminal is None:
        turns += 1
        msg = policy(messages, TOOL_SCHEMAS)
        if msg.get("role") != "assistant":
            msg = {"role": "assistant", "content": str(msg.get("content", "")), "tool_calls": msg.get("tool_calls", [])}
        messages.append(msg)
        calls = msg.get("tool_calls") or []
        if not calls:
            # A bare text reply is a protocol violation; nudge once and count it.
            parse_failures += 1
            messages.append({"role": "user", "content": "Please respond with a tool call (use done when finished)."})
            continue
        results_this_turn: list[ToolResult] = []
        for call in calls[:max_calls_per_turn]:
            fn = call.get("function", call)
            name = fn.get("name", "")
            args = fn.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args) if args.strip() else {}
                except json.JSONDecodeError:
                    parse_failures += 1
                    args = {"_raw": args}
            res = executor.execute(name, args if isinstance(args, dict) else {})
            results_this_turn.append(res)
            messages.append({"role": "tool", "name": name, "content": res.as_content()})
            if res.terminal and res.ok:
                terminal = name
                break
        if on_turn:
            on_turn(msg, results_this_turn)
    return Trajectory(
        messages=messages,
        tool_results=executor.log,
        terminal=terminal,
        final_house=house,
        turns=turns,
        parse_failures=parse_failures,
    )

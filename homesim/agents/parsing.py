"""Parse tool calls out of raw model text.

Handles Qwen's <tool_call>{...}</tool_call> blocks, Qwen3 <think> blocks, and a
fallback for bare JSON objects with name/arguments. Anything that fails to parse
is returned as free text so the episode runner can count it as a protocol failure.
"""

from __future__ import annotations

import json
import re
from typing import Any

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def _coerce_call(obj: Any) -> tuple[str, dict[str, Any]] | None:
    if not isinstance(obj, dict):
        return None
    name = obj.get("name") or (obj.get("function") or {}).get("name")
    args = obj.get("arguments", obj.get("parameters"))
    if args is None and "function" in obj:
        args = obj["function"].get("arguments")
    if not isinstance(name, str):
        return None
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {"_raw": args}
    if args is None:
        args = {}
    if not isinstance(args, dict):
        args = {"_raw": args}
    return name, args


def parse_assistant_text(text: str) -> dict[str, Any]:
    """Return an assistant message dict with `content` and `tool_calls`."""
    text = _THINK_RE.sub("", text).strip()
    calls: list[tuple[str, dict[str, Any]]] = []
    content_parts: list[str] = []
    last = 0
    for m in _TOOL_CALL_RE.finditer(text):
        content_parts.append(text[last:m.start()])
        last = m.end()
        try:
            call = _coerce_call(json.loads(m.group(1)))
        except json.JSONDecodeError:
            call = None
        if call:
            calls.append(call)
    content_parts.append(text[last:])
    content = " ".join(p.strip() for p in content_parts if p.strip())

    if not calls:
        # Fallback: a bare JSON object somewhere in the text.
        m = _JSON_OBJ_RE.search(text)
        if m:
            try:
                call = _coerce_call(json.loads(m.group(0)))
            except json.JSONDecodeError:
                call = None
            if call:
                calls.append(call)
                content = (text[:m.start()] + text[m.end():]).strip()
    return {
        "role": "assistant",
        "content": content,
        "tool_calls": [{"type": "function", "function": {"name": n, "arguments": a}} for n, a in calls],
    }

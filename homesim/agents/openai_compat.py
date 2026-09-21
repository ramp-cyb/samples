"""Policy that talks to any OpenAI-compatible chat endpoint (Ollama, vLLM, llama.cpp server).

Zero dependencies: uses urllib. Native tool calling is used when the server
returns `tool_calls`; otherwise the text is parsed for <tool_call> blocks.
"""

from __future__ import annotations

import json
import urllib.request
from typing import Any

from homesim.agents.parsing import parse_assistant_text


class OpenAICompatPolicy:
    def __init__(self, base_url: str = "http://localhost:11434/v1", model: str = "qwen3:1.7b",
                 api_key: str = "none", temperature: float = 0.0, max_tokens: int = 256, timeout: int = 120):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.last_raw: Any = None

    @staticmethod
    def _to_wire(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        wire = []
        for i, m in enumerate(messages):
            if m["role"] == "assistant" and m.get("tool_calls"):
                wire.append({
                    "role": "assistant",
                    "content": m.get("content") or "",
                    "tool_calls": [
                        {
                            "id": f"call_{i}_{j}",
                            "type": "function",
                            "function": {"name": c["function"]["name"], "arguments": json.dumps(c["function"]["arguments"])},
                        }
                        for j, c in enumerate(m["tool_calls"])
                    ],
                })
            elif m["role"] == "tool":
                # find the matching assistant call id: the most recent assistant message, nth tool result
                wire.append({"role": "tool", "content": m["content"], "name": m.get("name", "")})
            else:
                wire.append({"role": m["role"], "content": m.get("content") or ""})
        # attach tool_call_ids in order
        pending: list[str] = []
        for w in wire:
            if w["role"] == "assistant" and w.get("tool_calls"):
                pending = [c["id"] for c in w["tool_calls"]]
            elif w["role"] == "tool":
                w["tool_call_id"] = pending.pop(0) if pending else "call_0"
        return wire

    def __call__(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
        body = {
            "model": self.model,
            "messages": self._to_wire(messages),
            "tools": tools,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        self.last_raw = data
        msg = data["choices"][0]["message"]
        if msg.get("tool_calls"):
            calls = []
            for c in msg["tool_calls"]:
                fn = c["function"]
                args = fn.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args) if args.strip() else {}
                    except json.JSONDecodeError:
                        args = {"_raw": args}
                calls.append({"type": "function", "function": {"name": fn["name"], "arguments": args}})
            return {"role": "assistant", "content": msg.get("content") or "", "tool_calls": calls}
        return parse_assistant_text(msg.get("content") or "")

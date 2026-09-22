"""Exercise OpenAICompatPolicy against a stub OpenAI-compatible server.

This covers the wire format that Ollama, vLLM and llama.cpp speak, including
the tool_call_id pairing that strict servers reject when it is wrong.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from homesim import KnowledgeBase, generate_house, run_episode
from homesim.agents.openai_compat import OpenAICompatPolicy

RECEIVED: list[dict] = []
REPLIES: list[dict] = []


class StubHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        RECEIVED.append(body)
        message = REPLIES.pop(0) if REPLIES else {"content": "done", "tool_calls": None}
        payload = json.dumps({"choices": [{"message": message}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


@pytest.fixture
def server():
    RECEIVED.clear()
    REPLIES.clear()
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), StubHandler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def _native(name, args):
    return {"content": "", "tool_calls": [{"id": "x", "type": "function",
                                           "function": {"name": name, "arguments": json.dumps(args)}}]}


def test_native_tool_calls_drive_the_house(server):
    house = generate_house(4)
    house.set_device("living_room.tv", "power", "on")
    REPLIES.extend([
        _native("set_device", {"device_id": "living_room.tv", "attribute": "power", "value": "off"}),
        _native("done", {"message": "TV off."}),
    ])
    policy = OpenAICompatPolicy(base_url=server, model="stub")
    tr = run_episode(house, KnowledgeBase.for_house(house), policy, "Turn the telly off.", max_turns=5)
    assert tr.terminal == "done"
    assert house.devices["living_room.tv"].state["power"] == "off"
    assert [c["name"] for c in tr.calls] == ["set_device", "done"]


def test_text_tool_call_blocks_are_parsed(server):
    """A server that does not implement native tool calling still works."""
    house = generate_house(4)
    REPLIES.extend([
        {"content": 'Locking up.\n<tool_call>\n{"name": "set_device", "arguments": '
                    '{"device_id": "hallway.lock_front_door", "attribute": "locked", "value": true}}\n</tool_call>'},
        {"content": '<tool_call>{"name": "done", "arguments": {"message": "Locked."}}</tool_call>'},
    ])
    policy = OpenAICompatPolicy(base_url=server, model="stub")
    tr = run_episode(house, KnowledgeBase.for_house(house), policy, "Lock the front door.", max_turns=5)
    assert tr.terminal == "done"
    assert house.devices["hallway.lock_front_door"].state["locked"] is True


def test_wire_format_pairs_every_tool_result_with_its_call(server):
    house = generate_house(4)
    REPLIES.extend([
        _native("get_state", {}),
        _native("done", {"message": "All good."}),
    ])
    policy = OpenAICompatPolicy(base_url=server, model="stub")
    run_episode(house, KnowledgeBase.for_house(house), policy, "How are things?", max_turns=5)

    last = RECEIVED[-1]["messages"]
    assert last[0]["role"] == "system" and "tools" in RECEIVED[-1]
    ids = {}
    for m in last:
        if m["role"] == "assistant" and m.get("tool_calls"):
            for c in m["tool_calls"]:
                assert isinstance(c["function"]["arguments"], str)  # OpenAI sends arguments as a JSON string
                ids[c["id"]] = False
        elif m["role"] == "tool":
            assert m["tool_call_id"] in ids, "tool result references an unknown call id"
            ids[m["tool_call_id"]] = True
    assert ids and all(ids.values()), "every call must have a matching result"


def test_bare_text_reply_is_a_protocol_failure(server):
    house = generate_house(4)
    REPLIES.extend([{"content": "Sure, I'll get right on that."}] * 3)
    policy = OpenAICompatPolicy(base_url=server, model="stub")
    tr = run_episode(house, KnowledgeBase.for_house(house), policy, "Lock up.", max_turns=3)
    assert tr.terminal is None and tr.parse_failures == 3

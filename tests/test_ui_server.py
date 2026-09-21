import json
import threading
import time
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from homesim.ui.page import PAGE
from homesim.ui.server import Session, make_handler, oracle_factory


@pytest.fixture(scope="module")
def server():
    session = Session(seed=12, policy_factory=lambda: None, policy_label="oracle")
    session.policy_factory = oracle_factory(session)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(session, PAGE))
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", session
    httpd.shutdown()


def _get(base, path):
    return urllib.request.urlopen(base + path, timeout=5).read()


def _post(base, path, obj):
    req = urllib.request.Request(base + path, data=json.dumps(obj).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    return urllib.request.urlopen(req, timeout=5).read()


def test_serves_page_and_api(server):
    base, session = server
    assert b"<title>Home Simulator</title>" in _get(base, "/")
    intents = json.loads(_get(base, "/api/intents"))["intents"]
    assert "bedtime" in intents and "movie_mode" in intents
    state = json.loads(_get(base, "/api/state"))
    assert "rooms" in state and "hallway.thermostat" in {d["id"] for r in state["rooms"].values() for d in r["devices"]}
    docs = json.loads(_get(base, "/api/docs"))
    assert any(d["id"] == "scene_bedtime" for d in docs)


def test_episode_streams_events_and_changes_house(server):
    base, session = server
    events = []

    def listen():
        resp = urllib.request.urlopen(base + "/events", timeout=20)
        for line in resp:
            line = line.decode().strip()
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))
                if events[-1]["type"] == "episode_end":
                    return

    t = threading.Thread(target=listen, daemon=True)
    t.start()
    time.sleep(0.3)
    _post(base, "/api/run", {"intent": "bedtime", "delay": 0})
    t.join(timeout=20)

    kinds = [e["type"] for e in events]
    assert "user" in kinds and "turn" in kinds and kinds[-1] == "episode_end"
    end = events[-1]
    assert end["terminal"] == "done" and end["score"]["success"] is True
    assert all(d.state["locked"] for d in session.house.devices_of_kind("lock"))
    assert all(d.state["power"] == "off" for d in session.house.devices_of_kind("light"))


def test_reset_rebuilds_house(server):
    base, session = server
    _post(base, "/api/reset", {"seed": 77})
    assert session.seed == 77 and session.history == []


def test_free_text_without_intent_reports_error_for_oracle(server):
    base, session = server
    _post(base, "/api/reset", {"seed": 5})
    _post(base, "/api/run", {"utterance": "hello house", "intent": None, "delay": 0})
    for _ in range(50):
        time.sleep(0.1)
        errs = [h for h in session.history if h["type"] == "error"]
        if errs:
            assert "scenario" in errs[0]["message"]
            return
    raise AssertionError("expected an error event")

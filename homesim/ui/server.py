"""Live web UI: watch an episode run against the house, device by device.

Zero dependencies (stdlib http.server). Events stream to the browser over SSE.

  python scripts/serve_ui.py                       # oracle policy
  python scripts/serve_ui.py --policy openai --model qwen3:1.7b
"""

from __future__ import annotations

import json
import queue
import random
import threading
import traceback
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable
from urllib.parse import parse_qs, urlparse

from homesim.episode import run_episode
from homesim.evaluate import Expected, make_expected, score
from homesim.intents import INTENTS, sample_scenario
from homesim.knowledge import KnowledgeBase, generate_profile
from homesim.oracle import OraclePolicy, plan
from homesim.world import House, generate_house

PolicyFactory = Callable[[], Any]


@dataclass
class Session:
    """One house that episodes are run against, plus the event stream for the browser."""

    seed: int
    policy_factory: PolicyFactory
    policy_label: str
    house: House = field(init=False)
    kb: KnowledgeBase = field(init=False)
    subscribers: list[queue.Queue] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)
    running: bool = False
    history: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.reset(self.seed)

    def reset(self, seed: int) -> None:
        self.seed = seed
        self.house = generate_house(seed)
        self.kb = KnowledgeBase.for_house(self.house)
        self.history = []
        self.publish({"type": "reset", "house": self.house.snapshot(), "seed": seed,
                      "profile": generate_profile(seed).__dict__, "policy": self.policy_label})

    # -- event fan-out ------------------------------------------------------
    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue()
        with self.lock:
            self.subscribers.append(q)
        q.put({"type": "reset", "house": self.house.snapshot(), "seed": self.seed,
               "profile": generate_profile(self.seed).__dict__, "policy": self.policy_label})
        for h in self.history:
            q.put(h)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self.lock:
            if q in self.subscribers:
                self.subscribers.remove(q)

    def publish(self, event: dict[str, Any], remember: bool = False) -> None:
        if remember:
            self.history.append(event)
        with self.lock:
            subs = list(self.subscribers)
        for q in subs:
            q.put(event)

    # -- running an episode ---------------------------------------------------
    def run(self, utterance: str, intent: str | None, delay: float) -> None:
        if self.running:
            self.publish({"type": "error", "message": "an episode is already running"})
            return
        self.running = True
        try:
            rng = random.Random()
            sc = sample_scenario(self.house, rng, intent) if intent else None
            if sc is not None and not utterance:
                utterance = sc.utterance
            expected: Expected | None = None
            initial = self.house.device_states()
            if sc is not None:
                expected = make_expected(self.house, sc)
            self.publish({"type": "user", "text": utterance,
                          "scenario": sc.to_dict() if sc else None}, remember=True)

            def on_turn(msg: dict[str, Any], results: list[Any]) -> None:
                self.publish({
                    "type": "turn",
                    "content": msg.get("content") or "",
                    "calls": [
                        {
                            "name": r.name,
                            "arguments": r.arguments,
                            "ok": r.ok,
                            "changed": r.changed,
                            "result": r.result if r.name != "get_state" else "(house state)",
                        }
                        for r in results
                    ],
                    "house": self.house.snapshot(),
                }, remember=True)
                if delay:
                    threading.Event().wait(delay)

            policy = self.policy_factory()
            tr = run_episode(self.house, self.kb, policy, utterance, max_turns=20, on_turn=on_turn)
            summary: dict[str, Any] = {"type": "episode_end", "terminal": tr.terminal, "turns": tr.turns,
                                       "parse_failures": tr.parse_failures, "house": self.house.snapshot()}
            if expected is not None:
                summary["score"] = score(tr, expected, initial).to_dict()
                summary["expected_terminal"] = expected.terminal
            self.publish(summary, remember=True)
        except Exception as e:  # surfaced in the UI rather than killing the thread
            self.publish({"type": "error", "message": f"{type(e).__name__}: {e}",
                          "traceback": traceback.format_exc()[-1200:]}, remember=True)
        finally:
            self.running = False


def make_handler(session: Session, page: str):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):  # quiet
            pass

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            u = urlparse(self.path)
            if u.path == "/":
                self._send(200, page.encode("utf-8"), "text/html; charset=utf-8")
            elif u.path == "/api/intents":
                body = json.dumps({
                    "intents": sorted(d.name for d in INTENTS.values() if d.applies(session.house)),
                    "examples": {n: [t.text for t in INTENTS[n].templates[:3]] for n in INTENTS},
                }).encode()
                self._send(200, body, "application/json")
            elif u.path == "/api/state":
                self._send(200, json.dumps(session.house.snapshot()).encode(), "application/json")
            elif u.path == "/api/docs":
                body = json.dumps([d.to_dict() for d in session.kb.documents]).encode()
                self._send(200, body, "application/json")
            elif u.path == "/events":
                self.stream()
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self) -> None:
            u = urlparse(self.path)
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length) or b"{}")
            if u.path == "/api/run":
                threading.Thread(
                    target=session.run,
                    args=(payload.get("utterance", ""), payload.get("intent"), float(payload.get("delay", 0.35))),
                    daemon=True,
                ).start()
                self._send(200, b'{"ok":true}', "application/json")
            elif u.path == "/api/reset":
                seed = payload.get("seed")
                session.reset(int(seed) if seed not in (None, "") else random.randrange(1 << 20))
                self._send(200, b'{"ok":true}', "application/json")
            else:
                self._send(404, b"not found", "text/plain")

        def stream(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            q = session.subscribe()
            try:
                while True:
                    try:
                        event = q.get(timeout=15)
                        data = json.dumps(event)
                        self.wfile.write(f"data: {data}\n\n".encode("utf-8"))
                    except queue.Empty:
                        self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                session.unsubscribe(q)

    return Handler


def oracle_factory(session: Session) -> PolicyFactory:
    def factory():
        # The oracle needs a scenario; the UI re-samples one per run, so we rebuild
        # the plan lazily from the scenario attached to the last user event.
        scenario_event = next((h for h in reversed(session.history) if h["type"] == "user"), None)
        if not scenario_event or not scenario_event.get("scenario"):
            raise RuntimeError("the oracle policy needs a scenario: pick an intent rather than free text")
        from homesim.intents import Scenario

        sc = Scenario(**scenario_event["scenario"])
        return OraclePolicy(plan(session.house, sc))

    return factory


def serve(host: str, port: int, seed: int, policy: str, model: str, base_url: str, adapter: str | None) -> None:
    from homesim.ui.page import PAGE

    label = policy if policy == "oracle" else f"{policy}:{model}" + (" +lora" if adapter else "")
    session = Session(seed=seed, policy_factory=lambda: None, policy_label=label)
    if policy == "oracle":
        session.policy_factory = oracle_factory(session)
    elif policy == "openai":
        from homesim.agents.openai_compat import OpenAICompatPolicy

        shared = OpenAICompatPolicy(base_url=base_url, model=model)
        session.policy_factory = lambda: shared
    else:
        from homesim.agents.hf_policy import HFPolicy

        shared = HFPolicy(model, adapter=adapter)
        session.policy_factory = lambda: shared

    server = ThreadingHTTPServer((host, port), make_handler(session, PAGE))
    print(f"homesim UI on http://{host}:{port}  (policy: {label}, house seed {seed})")
    server.serve_forever()

"""The single-page dashboard served by homesim.ui.server (no build step, no CDN)."""

PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Home Simulator</title>
<style>
  :root {
    --bg: #f6f7f9; --panel: #ffffff; --ink: #14171f; --muted: #626b7d; --line: #e2e5ec;
    --accent: #3b6cf6; --ok: #1a8a5a; --warn: #b3761a; --bad: #c4392f; --chip: #f0f2f6;
    --flash: #fff2c9; --radius: 10px;
  }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      --bg: #12141a; --panel: #1a1d25; --ink: #e8eaf0; --muted: #98a1b3; --line: #2b3040;
      --accent: #6f95ff; --ok: #4cc38a; --warn: #e0a341; --bad: #f2695c; --chip: #232833;
      --flash: #3d3418;
    }
  }
  :root[data-theme="dark"] {
    --bg: #12141a; --panel: #1a1d25; --ink: #e8eaf0; --muted: #98a1b3; --line: #2b3040;
    --accent: #6f95ff; --ok: #4cc38a; --warn: #e0a341; --bad: #f2695c; --chip: #232833; --flash: #3d3418;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--bg); color: var(--ink);
    font: 14px/1.5 ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  }
  header {
    display: flex; flex-wrap: wrap; gap: 12px; align-items: center; justify-content: space-between;
    padding: 14px 16px; border-bottom: 1px solid var(--line); background: var(--panel);
  }
  h1 { font-size: 16px; margin: 0; letter-spacing: -0.01em; }
  .sub { color: var(--muted); font-size: 12px; }
  .wrap { display: grid; grid-template-columns: 1.25fr 1fr; gap: 16px; padding: 16px; align-items: start; }
  @media (max-width: 900px) { .wrap { grid-template-columns: 1fr; } }
  .card { background: var(--panel); border: 1px solid var(--line); border-radius: var(--radius); }
  .card h2 { font-size: 12px; text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted);
             margin: 0; padding: 12px 14px; border-bottom: 1px solid var(--line); }
  .pad { padding: 14px; }
  .rooms { display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 12px; padding: 14px; }
  .room { border: 1px solid var(--line); border-radius: var(--radius); padding: 10px 12px; background: var(--bg); }
  .room.occupied { border-color: var(--accent); }
  .room-head { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 8px; }
  .room-name { font-weight: 600; }
  .room-meta { font-size: 11px; color: var(--muted); }
  .dev { display: flex; align-items: center; gap: 8px; padding: 4px 0; font-size: 13px; border-radius: 6px; }
  .dev.flash { animation: flash 1.1s ease-out; }
  @keyframes flash { from { background: var(--flash); } to { background: transparent; } }
  .dot { width: 9px; height: 9px; border-radius: 50%; background: var(--line); flex: none; }
  .dot.on { background: var(--ok); } .dot.off { background: var(--muted); opacity: .45; }
  .dot.locked { background: var(--ok); } .dot.unlocked { background: var(--bad); }
  .dname { flex: 1; }
  .dval { color: var(--muted); font-variant-numeric: tabular-nums; font-size: 12px; }
  .bar { height: 4px; border-radius: 2px; background: var(--line); width: 46px; overflow: hidden; flex: none; }
  .bar > i { display: block; height: 100%; background: var(--accent); }
  .controls { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
  select, input[type=text], input[type=number], button {
    font: inherit; color: var(--ink); background: var(--panel);
    border: 1px solid var(--line); border-radius: 8px; padding: 7px 10px;
  }
  input[type=text] { flex: 1; min-width: 180px; }
  button { cursor: pointer; }
  button.primary { background: var(--accent); border-color: var(--accent); color: #fff; font-weight: 600; }
  button:disabled { opacity: .5; cursor: default; }
  .log { max-height: 60vh; overflow-y: auto; padding: 8px 14px 14px; }
  .entry { border-left: 2px solid var(--line); padding: 6px 0 6px 10px; margin: 8px 0; }
  .entry.user { border-color: var(--accent); }
  .entry.end { border-color: var(--ok); }
  .entry.error { border-color: var(--bad); }
  .role { font-size: 11px; text-transform: uppercase; letter-spacing: .06em; color: var(--muted); }
  .say { margin: 2px 0 6px; }
  .call { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px;
          background: var(--chip); border-radius: 6px; padding: 6px 8px; margin: 4px 0; word-break: break-word; }
  .call .fn { color: var(--accent); font-weight: 600; }
  .call.bad { outline: 1px solid var(--bad); }
  .tag { display: inline-block; font-size: 10px; padding: 1px 6px; border-radius: 99px;
         background: var(--chip); color: var(--muted); margin-left: 6px; }
  .tag.changed { color: var(--ok); } .tag.noop { color: var(--warn); } .tag.err { color: var(--bad); }
  table.kv { width: 100%; border-collapse: collapse; font-size: 12px; }
  table.kv td { padding: 3px 0; } table.kv td:last-child { text-align: right; color: var(--muted); font-variant-numeric: tabular-nums; }
  .pill { font-size: 11px; padding: 2px 8px; border-radius: 99px; background: var(--chip); }
  .pill.pass { color: var(--ok); } .pill.fail { color: var(--bad); }
  ul.lists { margin: 4px 0 0; padding-left: 18px; color: var(--muted); font-size: 12px; }
</style>
</head>
<body>
<header>
  <div>
    <h1>Home Simulator <span class="sub" id="policy"></span></h1>
    <div class="sub" id="clock">connecting…</div>
  </div>
  <div class="controls">
    <select id="intent"><option value="">free text…</option></select>
    <input type="text" id="utterance" placeholder="Say something to the house">
    <label class="sub">delay <input type="number" id="delay" value="0.35" step="0.05" min="0" max="3" style="width:70px"></label>
    <button class="primary" id="run">Send</button>
    <input type="number" id="seed" placeholder="seed" style="width:96px">
    <button id="reset">New house</button>
    <button id="theme" title="Toggle theme">◐</button>
  </div>
</header>

<div class="wrap">
  <div>
    <div class="card"><h2>House</h2><div class="rooms" id="rooms"></div></div>
    <div class="card" style="margin-top:16px"><h2>Lists &amp; schedule</h2><div class="pad" id="extras"></div></div>
  </div>
  <div>
    <div class="card"><h2>Episode</h2><div class="log" id="log"></div></div>
    <div class="card" style="margin-top:16px"><h2>Last score</h2><div class="pad" id="score"><span class="sub">no episode yet</span></div></div>
  </div>
</div>

<script>
const $ = (id) => document.getElementById(id);
let lastDevices = {};

function devDot(d) {
  if (d.kind === "lock") return d.locked ? "dot locked" : "dot unlocked";
  if ("power" in d) return d.power === "on" ? "dot on" : "dot off";
  if (d.kind === "blinds") return d.position === "open" ? "dot on" : "dot off";
  if (d.kind === "thermostat") return d.mode === "off" ? "dot off" : "dot on";
  return "dot";
}
function devValue(d) {
  if (d.kind === "light") return d.power === "on" ? `${d.brightness}% ${d.color_temp}` : "off";
  if (d.kind === "lock") return d.locked ? "locked" : "unlocked";
  if (d.kind === "blinds") return d.position;
  if (d.kind === "thermostat") return `${d.mode} · ${d.target}°`;
  if (d.kind === "tv" || d.kind === "speaker") return d.power === "on" ? `on · vol ${d.volume}` : "off";
  return d.power || "";
}
function renderHouse(h) {
  $("clock").textContent = `${h.time} · outside ${h.outside.temperature}° · ${h.outside.dark ? "dark" : "daylight"}`
    + (h.user_location ? ` · you are in the ${h.rooms[h.user_location] ? h.rooms[h.user_location].label : h.user_location}` : " · your location is unknown");
  const el = $("rooms");
  el.innerHTML = "";
  for (const [rid, room] of Object.entries(h.rooms)) {
    const div = document.createElement("div");
    div.className = "room" + (room.occupants.length ? " occupied" : "");
    div.innerHTML = `<div class="room-head"><span class="room-name">${room.label}</span>
      <span class="room-meta">${room.temperature}°${room.occupants.length ? " · " + room.occupants.join(", ") : ""}</span></div>`;
    for (const d of room.devices) {
      const prev = lastDevices[d.id];
      const changed = prev && JSON.stringify(prev) !== JSON.stringify(d);
      const row = document.createElement("div");
      row.className = "dev" + (changed ? " flash" : "");
      const bar = d.kind === "light"
        ? `<span class="bar"><i style="width:${d.brightness}%"></i></span>` : "";
      row.innerHTML = `<span class="${devDot(d)}"></span><span class="dname">${d.name}</span>${bar}<span class="dval">${devValue(d)}</span>`;
      div.appendChild(row);
      lastDevices[d.id] = d;
    }
    el.appendChild(div);
  }
  if (h.lists) {
    $("extras").innerHTML =
      `<strong>Shopping</strong><ul class="lists">${(h.lists.shopping.length ? h.lists.shopping : ["(empty)"]).map(i => `<li>${esc(i)}</li>`).join("")}</ul>`
    + `<strong style="display:block;margin-top:8px">Todo</strong><ul class="lists">${(h.lists.todo.length ? h.lists.todo : ["(empty)"]).map(i => `<li>${esc(i)}</li>`).join("")}</ul>`
    + `<strong style="display:block;margin-top:8px">Scheduled</strong><ul class="lists">${(h.schedule.length ? h.schedule.map(s => `${esc(s.device_id)} ${esc(s.attribute)}→${esc(String(s.value))} @ ${esc(s.when)}`) : ["(none)"]).map(i => `<li>${i}</li>`).join("")}</ul>`;
  }
}
function esc(s) { return String(s).replace(/[&<>]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c])); }
function addEntry(cls, html) {
  const d = document.createElement("div");
  d.className = "entry " + cls;
  d.innerHTML = html;
  $("log").appendChild(d);
  $("log").scrollTop = $("log").scrollHeight;
}
function renderCall(c) {
  const args = Object.entries(c.arguments || {}).map(([k, v]) => `${k}=${esc(JSON.stringify(v))}`).join(", ");
  let tag = "";
  if (!c.ok) tag = '<span class="tag err">error</span>';
  else if (["set_device", "create_schedule", "add_to_list"].includes(c.name)) tag = c.changed
    ? '<span class="tag changed">changed</span>' : '<span class="tag noop">no-op</span>';
  const err = (!c.ok) ? `<div class="sub">${esc(c.result)}</div>` : "";
  return `<div class="call${c.ok ? "" : " bad"}"><span class="fn">${esc(c.name)}</span>(${args})${tag}${err}</div>`;
}

const ev = new EventSource("/events");
ev.onmessage = (m) => {
  const e = JSON.parse(m.data);
  if (e.type === "reset") {
    lastDevices = {}; $("log").innerHTML = ""; $("score").innerHTML = '<span class="sub">no episode yet</span>';
    $("policy").textContent = "· policy: " + e.policy + " · seed " + e.seed;
    renderHouse(e.house);
    fetch("/api/intents").then(r => r.json()).then(d => {
      const sel = $("intent");
      const cur = sel.value;
      sel.innerHTML = '<option value="">free text…</option>' + d.intents.map(i => `<option value="${i}">${i}</option>`).join("");
      sel.value = cur;
    });
  } else if (e.type === "user") {
    addEntry("user", `<div class="role">user${e.scenario ? " · intent: " + esc(e.scenario.intent) : ""}</div><div class="say">${esc(e.text)}</div>`);
  } else if (e.type === "turn") {
    addEntry("", `<div class="role">assistant</div>${e.content ? `<div class="say">${esc(e.content)}</div>` : ""}${(e.calls || []).map(renderCall).join("")}`);
    renderHouse(e.house);
  } else if (e.type === "episode_end") {
    addEntry("end", `<div class="role">finished</div><div class="say">${esc(e.terminal || "no terminal call")} after ${e.turns} turns</div>`);
    renderHouse(e.house);
    if (e.score) {
      const s = e.score;
      const pill = (ok, label) => `<span class="pill ${ok ? "pass" : "fail"}">${label}</span>`;
      $("score").innerHTML = `<div style="margin-bottom:8px">${pill(s.success, s.success ? "success" : "failed")}
        ${pill(s.terminal_correct, "terminal " + (s.terminal_correct ? "ok" : "wrong"))}
        ${pill(s.state_match, "state " + (s.state_match ? "match" : "mismatch"))}
        ${s.unsafe ? pill(false, "UNSAFE") : ""}</div>
        <table class="kv">
          <tr><td>tool validity</td><td>${(s.tool_validity * 100).toFixed(0)}%</td></tr>
          <tr><td>invalid calls</td><td>${s.invalid_calls}</td></tr>
          <tr><td>redundant calls</td><td>${s.redundant_calls}</td></tr>
          <tr><td>missing changes</td><td>${s.missing_changes}</td></tr>
          <tr><td>extra changes</td><td>${s.extra_changes}</td></tr>
          <tr><td>searched context</td><td>${s.searched ? "yes" : "no"}${s.search_expected ? " (expected)" : ""}</td></tr>
          <tr><td>parse failures</td><td>${s.parse_failures}</td></tr>
          <tr><td>turns</td><td>${s.turns}</td></tr>
        </table>`;
    } else {
      $("score").innerHTML = '<span class="sub">free-text episode: no oracle target to score against</span>';
    }
    $("run").disabled = false;
  } else if (e.type === "error") {
    addEntry("error", `<div class="role">error</div><div class="say">${esc(e.message)}</div>`);
    $("run").disabled = false;
  }
};

$("run").onclick = () => {
  $("run").disabled = true;
  fetch("/api/run", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ utterance: $("utterance").value, intent: $("intent").value || null, delay: parseFloat($("delay").value || "0") }),
  });
  $("utterance").value = "";
};
$("utterance").addEventListener("keydown", (e) => { if (e.key === "Enter") $("run").click(); });
$("reset").onclick = () => fetch("/api/reset", {
  method: "POST", headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ seed: $("seed").value }),
});
$("theme").onclick = () => {
  const cur = document.documentElement.getAttribute("data-theme");
  document.documentElement.setAttribute("data-theme", cur === "dark" ? "light" : "dark");
};
</script>
</body>
</html>
"""

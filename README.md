# homesim

A simulated smart home for training and evaluating small local language models on
**tool calling**. The user speaks plain English. The model has to work out what they
mean, retrieve the household's own conventions, and drive the house through tool
calls. A rule-coded oracle defines the correct behaviour, generates the training
data, and provides the evaluation target.

The experiment it is built for: **a 1B–3B model fine-tuned on a few thousand
synthetic episodes should beat a much larger prompted model at this task.**

```
pip install -e ".[test]"
pytest                                  # 447 tests
python scripts/serve_ui.py              # live dashboard on http://127.0.0.1:8000
```

## Why it is not a trivial mapping

The user never names a device or an action:

| They say | It means |
| --- | --- |
| "Popcorn's ready, we're putting a movie on." | dim the living room lights to *this household's* movie level, warm colour, close the blinds, TV on, speaker on at the stored volume |
| "The milk's gone off." | add milk to the shopping list, touch no device |
| "Let me in through the front door." at 23:00 | refuse and ask for confirmation in the app, because of a house rule |
| "It's too dark in here." with the user's room unknown | ask which room |
| "Grandma's visiting on Friday." | schedule the guest room heater for Friday at the stored warm-up time, add the guest bed to the todo list |

Scene parameters differ per house and live only in the knowledge base, so the model
cannot memorise them. It has to call `search_context`, read the document and apply it.

## Architecture

```
world.py       House, rooms, devices, deterministic generator from a seed
knowledge.py   per-house documents (scenes, preferences, rules) + BM25 retrieval
tools.py       the 7 tools the model gets, and their executor
intents.py     18 intent classes x 10 English templates each, no tool vocabulary
oracle.py      rule-coded agent: spec + house -> the ideal trajectory
episode.py     the policy <-> house loop
evaluate.py    scoring against the oracle's target state
datagen.py     chat-format JSONL with train / eval splits
agents/        model policies: local transformers, or any OpenAI-compatible server
ui/            live browser dashboard (stdlib only, SSE)
```

### The tools

`search_context` · `get_state` · `set_device` · `create_schedule` · `add_to_list` ·
`ask_user` · `done`

Every episode ends with exactly one `done` or `ask_user`.

## Workflow

**1. Generate data.** Held-out evaluation uses unseen houses *and* unseen English
phrasings, so in-distribution and generalisation are reported separately.

```
python scripts/generate_data.py --train-houses 400 --per-house 6
# data/train.jsonl, data/eval_in_dist.jsonl, data/eval_heldout_phrasing.jsonl
```

**2. Measure the untuned baseline.** Run a local model through Ollama or vLLM, or
straight from transformers.

```
python scripts/run_eval.py --policy openai --model qwen3:1.7b --data data/eval_in_dist.jsonl
```

**3. Fine-tune.** LoRA, loss masked to assistant tokens. Needs a GPU; a 1.7B adapter
over a few thousand episodes trains in minutes.

```
pip install -e ".[train]"
python scripts/train_lora.py --model Qwen/Qwen3-1.7B --data data/train.jsonl --out checkpoints/lora
```

**4. Compare.**

```
python scripts/benchmark.py --config '[
  {"label":"base 1.7b",  "policy":"openai","model":"qwen3:1.7b"},
  {"label":"tuned 1.7b", "policy":"hf","model":"Qwen/Qwen3-1.7B","adapter":"checkpoints/lora"},
  {"label":"base 7b",    "policy":"openai","model":"qwen3:8b"}]'
```

## Metrics

Outcome is judged by the **final state of the house**, not by matching the oracle's
call sequence, so a different-but-correct route still passes.

| Metric | Meaning |
| --- | --- |
| `success` | right terminal call, house in the target state, nothing unsafe |
| `state_match` | devices, lists and schedule all equal the oracle's result |
| `terminal_correct` | ended with `done` vs `ask_user` as required |
| `tool_validity` | fraction of calls that executed without an error |
| `invalid_calls` | bad device ids, wrong attributes, out-of-range values |
| `redundant_calls` | state-changing calls that changed nothing |
| `unsafe` | unlocked a door inside the night-time rule window |
| `parse_failures` | replies with no parseable tool call |

The oracle scores 1.000 on every split by construction, which is the suite's own
sanity check (`tests/test_datagen_eval.py::test_oracle_scores_perfectly`).

## Live UI

`python scripts/serve_ui.py` opens a dashboard showing the floorplan, every device's
state, and the transcript of tool calls as they execute. Devices flash when they
change. Point it at a real model to watch it work, or fail:

```
python scripts/serve_ui.py --policy openai --model qwen3:1.7b
python scripts/serve_ui.py --policy hf --model Qwen/Qwen3-1.7B --adapter checkpoints/lora
```

Picking an intent from the dropdown scores the episode against the oracle. Typing
free text runs the model unscored.

## Model choice

Qwen models are the default because their chat template has native tool-call
support, so the prompted baseline and the fine-tune use the same format and the
comparison is clean. Gemma and Llama 3.2 at this size lack a tool template, which
muddies it. `agents/parsing.py` handles `<tool_call>` blocks, `<think>` blocks and
bare JSON, so a model that improvises is parsed rather than silently failed.

import pytest

from homesim.datagen import generate_split
from homesim.training import (
    TemplateMismatch,
    build_examples,
    build_segments,
    segments_to_example,
)
from tests.fake_tokenizer import IM_END, IM_START, FakeQwenTokenizer


@pytest.fixture(scope="module")
def records():
    return list(generate_split("train", 4, 3, seed=11))


def test_segments_reconstruct_the_full_render(records):
    tok = FakeQwenTokenizer()
    for rec in records:
        segments = build_segments(rec["messages"], rec["tools"], tok)
        joined = "".join(s.text for s in segments)
        full = tok.apply_chat_template(rec["messages"], tools=rec["tools"], add_generation_prompt=False, tokenize=False)
        assert joined == full


def test_trainable_spans_are_exactly_the_assistant_turns(records):
    tok = FakeQwenTokenizer()
    rec = records[0]
    segments = build_segments(rec["messages"], rec["tools"], tok)
    targets = [s.text for s in segments if s.trainable]
    assistants = [m for m in rec["messages"] if m["role"] == "assistant"]
    assert len(targets) == len(assistants)
    for text, msg in zip(targets, assistants):
        # A target span is the assistant body plus the end token, and never the
        # "<|im_start|>assistant" header, which belongs to the generation prompt.
        assert not text.startswith(IM_START)
        assert text.endswith(IM_END + "\n")
        for call in msg.get("tool_calls", []):
            assert f'"name": "{call["function"]["name"]}"' in text


def test_labels_cover_only_trainable_text(records):
    tok = FakeQwenTokenizer()
    rec = records[0]
    segments = build_segments(rec["messages"], rec["tools"], tok)
    ex = segments_to_example(segments, tok)
    assert len(ex["input_ids"]) == len(ex["labels"])
    unmasked = "".join(chr(l) for l in ex["labels"] if l != -100)
    assert unmasked == "".join(s.text for s in segments if s.trainable)
    # The system prompt carries the house state and must never be a target.
    assert "House state:" not in unmasked


def test_prompt_prefix_matches_inference_rendering(records):
    """What the model is asked to continue at training time must equal what
    HFPolicy sends at inference time."""
    tok = FakeQwenTokenizer()
    rec = records[0]
    messages = rec["messages"]
    segments = build_segments(messages, rec["tools"], tok)
    first_assistant = next(i for i, m in enumerate(messages) if m["role"] == "assistant")
    inference_prompt = tok.apply_chat_template(
        messages[:first_assistant], tools=rec["tools"], add_generation_prompt=True, tokenize=False
    )
    assert segments[0].text == inference_prompt
    assert segments[0].trainable is False


def test_non_additive_template_is_rejected(records):
    tok = FakeQwenTokenizer(thinking_rewrites_history=True)
    rec = next(r for r in records if sum(m["role"] == "assistant" for m in r["messages"]) >= 2)
    messages = [dict(m) for m in rec["messages"]]
    for m in messages:
        if m["role"] == "assistant":
            m["content"] = "<think>planning</think>" + (m.get("content") or "")
    with pytest.raises(TemplateMismatch):
        build_segments(messages, rec["tools"], tok)


def test_build_examples_stats_and_length_filter(records):
    tok = FakeQwenTokenizer()
    examples, stats = build_examples(records, tok, max_len=10**9)
    assert stats.kept == len(records) and stats.dropped_too_long == 0
    assert stats.max_tokens > 0 and stats.mean_tokens > 0
    assert all(any(l != -100 for l in e["labels"]) for e in examples)

    examples, stats = build_examples(records, tok, max_len=10)
    assert examples == [] and stats.dropped_too_long == len(records)
    assert "dropped as too long" in stats.describe()

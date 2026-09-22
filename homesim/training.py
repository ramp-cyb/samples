"""Turn episode records into token-level training examples.

The tricky part is the label mask. We want the loss on exactly the assistant
turns and nothing else, while the text the model sees at training time must be
byte-identical to what `apply_chat_template(..., add_generation_prompt=True)`
produces at inference time. Otherwise the model trains on a format it will
never be prompted with.

The approach: render the conversation prefix twice at each assistant turn, once
without that turn and once with it, and take the string difference. The chat
template itself renders the assistant turn, including its tool calls, so the
training format is whatever the model's own template produces. Segments are
then tokenized in order and concatenated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


class Tokenizer(Protocol):
    def apply_chat_template(self, messages: list[dict[str, Any]], **kwargs: Any) -> str: ...
    def __call__(self, text: str, **kwargs: Any) -> dict[str, list[int]]: ...


class TemplateMismatch(RuntimeError):
    """The chat template does not extend its own prefix, so spans cannot be located."""


@dataclass
class Segment:
    text: str
    trainable: bool


def build_segments(messages: list[dict[str, Any]], tools: list[dict[str, Any]], tokenizer: Tokenizer,
                   **template_kwargs: Any) -> list[Segment]:
    """Split the rendered conversation into alternating prompt and assistant spans.

    Concatenating the segment texts reproduces the full rendered conversation exactly.
    """
    def render(msgs: list[dict[str, Any]], generation_prompt: bool) -> str:
        return tokenizer.apply_chat_template(
            msgs, tools=tools, add_generation_prompt=generation_prompt, tokenize=False, **template_kwargs
        )

    segments: list[Segment] = []
    cursor = 0
    consumed = ""
    for i, msg in enumerate(messages):
        if msg["role"] != "assistant":
            continue
        prefix_text = render(messages[:i], generation_prompt=True)
        full_text = render(messages[: i + 1], generation_prompt=False)
        if not prefix_text.startswith(consumed):
            raise TemplateMismatch(
                f"message {i}: rendering the prefix did not extend the previous render. "
                "The chat template rewrites earlier turns, so assistant spans cannot be located."
            )
        if not full_text.startswith(prefix_text):
            raise TemplateMismatch(
                f"message {i}: the assistant turn does not begin with the generation prompt. "
                "Training and inference formats would differ."
            )
        prompt_span = prefix_text[cursor:]
        if prompt_span:
            segments.append(Segment(prompt_span, trainable=False))
        target_span = full_text[len(prefix_text):]
        if target_span:
            segments.append(Segment(target_span, trainable=True))
        cursor = len(full_text)
        consumed = full_text

    final_text = render(messages, generation_prompt=False)
    if not final_text.startswith(consumed):
        raise TemplateMismatch("the final render does not extend the last assistant turn")
    tail = final_text[cursor:]
    if tail:
        segments.append(Segment(tail, trainable=False))
    return segments


def segments_to_example(segments: list[Segment], tokenizer: Tokenizer) -> dict[str, list[int]]:
    """Tokenize segments in order, masking labels outside the trainable spans."""
    input_ids: list[int] = []
    labels: list[int] = []
    for seg in segments:
        ids = tokenizer(seg.text, add_special_tokens=False)["input_ids"]
        input_ids.extend(ids)
        labels.extend(ids if seg.trainable else [-100] * len(ids))
    return {"input_ids": input_ids, "labels": labels}


@dataclass
class BuildStats:
    kept: int = 0
    dropped_too_long: int = 0
    dropped_no_target: int = 0
    max_tokens: int = 0
    total_tokens: int = 0

    @property
    def mean_tokens(self) -> float:
        return self.total_tokens / self.kept if self.kept else 0.0

    def describe(self) -> str:
        return (
            f"{self.kept} examples kept, {self.dropped_too_long} dropped as too long, "
            f"{self.dropped_no_target} dropped with no trainable tokens; "
            f"mean {self.mean_tokens:.0f} tokens, max {self.max_tokens}"
        )


def build_examples(records: list[dict[str, Any]], tokenizer: Tokenizer, max_len: int,
                   **template_kwargs: Any) -> tuple[list[dict[str, list[int]]], BuildStats]:
    """One training example per episode, with loss masked to assistant turns."""
    examples: list[dict[str, list[int]]] = []
    stats = BuildStats()
    for rec in records:
        segments = build_segments(rec["messages"], rec["tools"], tokenizer, **template_kwargs)
        example = segments_to_example(segments, tokenizer)
        n = len(example["input_ids"])
        if n > max_len:
            stats.dropped_too_long += 1
            continue
        if not any(l != -100 for l in example["labels"]):
            stats.dropped_no_target += 1
            continue
        stats.kept += 1
        stats.total_tokens += n
        stats.max_tokens = max(stats.max_tokens, n)
        examples.append(example)
    return examples, stats

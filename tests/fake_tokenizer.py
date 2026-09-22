"""A Qwen-shaped chat template and tokenizer, good enough to test span masking.

Mirrors the real Qwen ChatML format: tools in the system turn, assistant tool
calls inside <tool_call> blocks, tool results wrapped as <tool_response> in a
user turn, and a generation prompt of "<|im_start|>assistant\n".
"""

from __future__ import annotations

import json
import re
from typing import Any

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"


class FakeQwenTokenizer:
    """Additive template: rendering n+1 messages always extends rendering n."""

    eos_token = IM_END

    def __init__(self, thinking_rewrites_history: bool = False):
        # When True, the template strips <think> from earlier assistant turns,
        # which breaks the additive property. Used to test the guard.
        self.thinking_rewrites_history = thinking_rewrites_history

    def apply_chat_template(self, messages: list[dict[str, Any]], tools=None, add_generation_prompt=False,
                            tokenize=False, **kwargs) -> str:
        assert tokenize is False, "tests only use the string form"
        out = []
        for i, m in enumerate(messages):
            role = m["role"]
            if role == "system":
                body = m["content"]
                if tools:
                    body += "\n\n# Tools\n" + "\n".join(json.dumps(t) for t in tools)
                out.append(f"{IM_START}system\n{body}{IM_END}\n")
            elif role == "user":
                out.append(f"{IM_START}user\n{m['content']}{IM_END}\n")
            elif role == "tool":
                out.append(f"{IM_START}user\n<tool_response>\n{m['content']}\n</tool_response>{IM_END}\n")
            elif role == "assistant":
                parts = []
                content = m.get("content") or ""
                is_last = i == len(messages) - 1
                if self.thinking_rewrites_history and not is_last:
                    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
                if content:
                    parts.append(content)
                for c in m.get("tool_calls", []):
                    fn = c["function"]
                    payload = json.dumps({"name": fn["name"], "arguments": fn["arguments"]}, ensure_ascii=False)
                    parts.append(f"<tool_call>\n{payload}\n</tool_call>")
                out.append(f"{IM_START}assistant\n" + "\n".join(parts) + f"{IM_END}\n")
            else:
                raise ValueError(role)
        text = "".join(out)
        if add_generation_prompt:
            text += f"{IM_START}assistant\n"
        return text

    def __call__(self, text: str, add_special_tokens: bool = True, **kwargs) -> dict[str, list[int]]:
        # Character-level ids keep token boundaries aligned with string slices,
        # which is what makes the mask assertions in the tests exact.
        return {"input_ids": [ord(c) for c in text]}

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(i) for i in ids)

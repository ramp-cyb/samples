"""Policy backed by a local transformers model (optionally with a LoRA adapter).

Requires the `model` extra: pip install -e ".[model]"  (and `peft` for adapters).
"""

from __future__ import annotations

from typing import Any

from homesim.agents.parsing import parse_assistant_text


class HFPolicy:
    def __init__(self, model_name: str, adapter: str | None = None, max_new_tokens: int = 256,
                 device_map: str = "auto", dtype: str | None = None, temperature: float = 0.0,
                 enable_thinking: bool = False):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        torch_dtype = getattr(torch, dtype) if dtype else (torch.bfloat16 if torch.cuda.is_available() else torch.float32)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device_map, torch_dtype=torch_dtype)
        if adapter:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model.eval()
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.enable_thinking = enable_thinking
        self.last_raw: str = ""

    def render(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> str:
        kwargs: dict[str, Any] = {"tools": tools, "add_generation_prompt": True, "tokenize": False}
        # Qwen3 accepts enable_thinking; other templates ignore unknown kwargs.
        kwargs["enable_thinking"] = self.enable_thinking
        return self.tokenizer.apply_chat_template(messages, **kwargs)

    def __call__(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        prompt = self.render(messages, tools)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        gen_kwargs: dict[str, Any] = {"max_new_tokens": self.max_new_tokens, "pad_token_id": self.tokenizer.eos_token_id}
        if self.temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=self.temperature)
        else:
            gen_kwargs["do_sample"] = False
        with torch.no_grad():
            out = self.model.generate(**inputs, **gen_kwargs)
        text = self.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        self.last_raw = text
        return parse_assistant_text(text)

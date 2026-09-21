#!/usr/bin/env python3
"""LoRA fine-tune a small instruct model on the generated trajectories.

Runs on a single GPU (a 1.7B LoRA over a few thousand episodes takes minutes).

  pip install -e ".[train]"
  python scripts/train_lora.py --model Qwen/Qwen3-1.7B --data data/train.jsonl --out checkpoints/lora

Loss is masked to assistant tokens only, so the model learns to produce tool
calls and never to predict the system prompt or the tool results.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
from typing import Any

TOOL_CALL_TEMPLATE = "<tool_call>\n{payload}\n</tool_call>"


def render_assistant(msg: dict[str, Any]) -> str:
    parts = []
    if msg.get("content"):
        parts.append(msg["content"])
    for c in msg.get("tool_calls", []):
        payload = json.dumps({"name": c["function"]["name"], "arguments": c["function"]["arguments"]}, ensure_ascii=False)
        parts.append(TOOL_CALL_TEMPLATE.format(payload=payload))
    return "\n".join(parts)


def build_examples(records: list[dict[str, Any]], tokenizer, max_len: int) -> list[dict[str, list[int]]]:
    """One training example per episode; labels masked to assistant spans."""
    examples = []
    for rec in records:
        messages = rec["messages"]
        input_ids: list[int] = []
        labels: list[int] = []
        prefix: list[dict[str, Any]] = []
        for msg in messages:
            if msg["role"] == "assistant":
                rendered = tokenizer.apply_chat_template(
                    prefix, tools=rec["tools"], add_generation_prompt=True, tokenize=False
                )
                prompt_ids = tokenizer(rendered, add_special_tokens=False)["input_ids"]
                new = prompt_ids[len(input_ids):]
                input_ids.extend(new)
                labels.extend([-100] * len(new))
                target = render_assistant(msg) + tokenizer.eos_token
                target_ids = tokenizer(target, add_special_tokens=False)["input_ids"]
                input_ids.extend(target_ids)
                labels.extend(target_ids)
            prefix.append(msg)
        if len(input_ids) > max_len:
            continue
        examples.append({"input_ids": input_ids, "labels": labels})
    return examples


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--data", default="data/train.jsonl")
    ap.add_argument("--out", default="checkpoints/lora")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--alpha", type=int, default=64)
    args = ap.parse_args()

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    from homesim.datagen import read_jsonl

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    records = read_jsonl(args.data)
    examples = build_examples(records, tokenizer, args.max_len)
    print(f"{len(examples)} training examples (dropped {len(records) - len(examples)} over {args.max_len} tokens)")

    def collate(batch):
        maxlen = max(len(b["input_ids"]) for b in batch)
        pad = tokenizer.pad_token_id or tokenizer.eos_token_id
        input_ids, labels, mask = [], [], []
        for b in batch:
            n = maxlen - len(b["input_ids"])
            input_ids.append(b["input_ids"] + [pad] * n)
            labels.append(b["labels"] + [-100] * n)
            mask.append([1] * len(b["input_ids"]) + [0] * n)
        return {
            "input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(mask),
        }

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32, device_map="auto"
    )
    model.config.use_cache = False
    peft_config = LoraConfig(
        r=args.rank, lora_alpha=args.alpha, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=args.out,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            logging_steps=10,
            save_strategy="epoch",
            bf16=torch.cuda.is_available(),
            lr_scheduler_type="cosine",
            warmup_ratio=0.03,
            report_to=[],
        ),
        train_dataset=examples,
        data_collator=collate,
    )
    trainer.train()
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"adapter saved to {args.out}")


if __name__ == "__main__":
    main()

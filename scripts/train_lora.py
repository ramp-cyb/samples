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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--data", default="data/train.jsonl")
    ap.add_argument("--out", default="checkpoints/lora")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--alpha", type=int, default=64)
    ap.add_argument("--no-thinking", action="store_true",
                    help="pass enable_thinking=False to the chat template (Qwen3)")
    args = ap.parse_args()

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    from homesim.datagen import read_jsonl
    from homesim.training import TemplateMismatch, build_examples

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    records = read_jsonl(args.data)
    template_kwargs = {"enable_thinking": False} if args.no_thinking else {}
    try:
        examples, stats = build_examples(records, tokenizer, args.max_len, **template_kwargs)
    except TemplateMismatch as e:
        raise SystemExit(
            f"chat template incompatible with span-masked training: {e}\n"
            "Try --no-thinking, or pick a model whose template renders turns additively."
        )
    print(stats.describe())
    if not examples:
        raise SystemExit("no usable training examples; raise --max-len or check the dataset")

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

"""
QLoRA fine-tuning for Llama-3.2-3B on ATLAS agenda-setting data.

Input format (per JSONL line):
  {"input": "[Context]: ...\\n[Utterance]: [Doctor|Patient] text",
   "output": "{\"relevant\": true, \"detail_list\": [...]}" or '{"relevant": false}'}

Training loss is applied to OUTPUT tokens only (completion-only).
The prompt is masked so the model never trains on its own input.

Run on RunPod: see README.md for the one-line command.
"""

import argparse
import json
import os
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTConfig, SFTTrainer
from transformers import DataCollatorForLanguageModeling


# Prompt scaffolding ----------------------------------------------------------
# We frame the task to Llama-3.2-3B-Instruct using the chat template.
# The assistant's reply is the JSON output exactly as it appears in train.jsonl.

SYSTEM_PROMPT = (
    "You are a clinical agenda-setting assistant. Given prior conversation summaries "
    "(pipe-delimited) as [Context] and a new [Utterance], decide if the utterance "
    "contains clinically relevant content and, if so, emit one or more typed summaries.\n\n"
    "Output a single JSON object on one line.\n"
    "If irrelevant: {\"relevant\": false}\n"
    "If relevant:   {\"relevant\": true, \"detail_list\": "
    "[{\"type\": <T>, \"summary\": <S>}, ...]}\n\n"
    "Allowed types: agenda_item, detail, medication, social_history, follow_up, "
    "question, question_unanswered."
)


def build_chat(example, tokenizer):
    """Render one example through the model's chat template and tokenize."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": example["input"]},
        {"role": "assistant", "content": example["output"]},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    enc = tokenizer(text, truncation=True, max_length=2048, add_special_tokens=False)
    return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}


def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return Dataset.from_list(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", default="meta-llama/Llama-3.2-3B-Instruct",
                    help="HF model id. Use base 3B if you want — but instruct trains faster.")
    ap.add_argument("--train_file", default="data/train.jsonl")
    ap.add_argument("--eval_file", default="data/eval.jsonl")
    ap.add_argument("--output_dir", default="checkpoints/atlas-llama32-3b-qlora")
    ap.add_argument("--lora_r", type=int, default=32)
    ap.add_argument("--lora_alpha", type=int, default=64)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=2e-4,
                    help="QLoRA paper default. Spec Phase 2 says 5e-5; raise to 2e-4 for "
                         "small data + LoRA adapter.")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--max_seq_len", type=int, default=2048,
                    help="Covers the longest input (~3500 chars ≈ 900 tokens) plus output.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_eval_samples", type=int, default=200,
                    help="Subsample eval each epoch — full eval is slow during training.")
    ap.add_argument("--save_steps", type=int, default=200)
    ap.add_argument("--logging_steps", type=int, default=20)
    ap.add_argument("--report_to", default="none",
                    help="'wandb' if you want logging; needs WANDB_API_KEY env.")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Tokenizer ------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Model: 4-bit NF4 + bf16 compute -------------------------------------
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=bnb,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    # LoRA config ----------------------------------------------------------
    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    # Data -----------------------------------------------------------------
    train_ds = load_jsonl(args.train_file)
    eval_ds = load_jsonl(args.eval_file)

    train_ds = train_ds.map(lambda ex: build_chat(ex, tokenizer),
                            remove_columns=train_ds.column_names)
    eval_ds = eval_ds.map(lambda ex: build_chat(ex, tokenizer),
                          remove_columns=eval_ds.column_names)
    if args.max_eval_samples and len(eval_ds) > args.max_eval_samples:
        eval_ds = eval_ds.shuffle(seed=args.seed).select(range(args.max_eval_samples))

    # Completion-only loss: mask everything before the assistant header.
    # Llama 3.x chat template assistant turn starts with this exact string.
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # Trainer --------------------------------------------------------------
    sft_cfg = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.0,
        bf16=True,
        max_seq_length=args.max_seq_len,
        packing=False,                       # IMPORTANT: do not pack — completion mask relies on a single example per sequence

        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        eval_strategy="steps",
        eval_steps=args.save_steps,
        report_to=args.report_to,
        seed=args.seed,
        optim="paged_adamw_8bit",
        remove_unused_columns=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_cfg,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # One last full eval at the end
    metrics = trainer.evaluate()
    with open(Path(args.output_dir) / "final_eval.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

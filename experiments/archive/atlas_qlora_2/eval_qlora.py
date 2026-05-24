"""
Evaluate a QLoRA-fine-tuned Llama-3.2-3B against eval.jsonl.

Metrics computed:
  - JSON parse rate
  - Relevance accuracy (binary)
  - Type detection (per-class precision/recall/F1, macro-F1) — only on relevant examples
  - Summary BERTScore (F1) when both pred and gold detail_list have a summary

This script does NOT compute system-level metrics (linking accuracy, first-mention,
resolution) — those require the full harness (state_tracker.py).
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from train_qlora import SYSTEM_PROMPT  # reuse the exact same prompt

VALID_TYPES = [
    "agenda_item", "detail", "medication", "social_history",
    "follow_up", "question", "question_unanswered",
]


def load_model(base_id: str, adapter_dir: str):
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tok = AutoTokenizer.from_pretrained(adapter_dir or base_id, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        base_id,
        quantization_config=bnb,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    if adapter_dir:
        model = PeftModel.from_pretrained(base, adapter_dir)
    else:
        model = base
    model.eval()
    return model, tok


@torch.no_grad()
def generate(model, tok, user_text: str, max_new_tokens: int = 256) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.0,
        pad_token_id=tok.eos_token_id,
    )
    text = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return text.strip()


JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_output(raw: str):
    """Best-effort JSON parse of model output."""
    raw = raw.strip()
    # Strip markdown fences if any
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    try:
        return json.loads(raw), True
    except Exception:
        m = JSON_RE.search(raw)
        if m:
            try:
                return json.loads(m.group(0)), True
            except Exception:
                pass
    return None, False


def first_type(parsed) -> str:
    if not parsed or not parsed.get("relevant"):
        return "irrelevant"
    dl = parsed.get("detail_list") or []
    if not dl:
        return "irrelevant"
    return dl[0].get("type", "irrelevant")


def first_summary(parsed):
    if not parsed or not parsed.get("relevant"):
        return None
    dl = parsed.get("detail_list") or []
    if not dl:
        return None
    return dl[0].get("summary")


def macro_f1(pred_types, gold_types, classes):
    tp, fp, fn = Counter(), Counter(), Counter()
    for p, g in zip(pred_types, gold_types):
        if p == g:
            tp[p] += 1
        else:
            fp[p] += 1
            fn[g] += 1
    per_class = {}
    f1s = []
    for c in classes:
        p = tp[c] / max(tp[c] + fp[c], 1)
        r = tp[c] / max(tp[c] + fn[c], 1)
        f1 = 2 * p * r / max(p + r, 1e-8)
        per_class[c] = {"precision": p, "recall": r, "f1": f1, "support": tp[c] + fn[c]}
        f1s.append(f1)
    return per_class, sum(f1s) / max(len(f1s), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="meta-llama/Llama-3.2-3B-Instruct")
    ap.add_argument("--adapter", default="checkpoints/atlas-llama32-3b-qlora")
    ap.add_argument("--eval_file", default="data/eval.jsonl")
    ap.add_argument("--limit", type=int, default=None,
                    help="Cap eval examples for a quick smoke test.")
    ap.add_argument("--bertscore", action="store_true",
                    help="Compute BERTScore on summaries (slow; needs internet).")
    ap.add_argument("--output", default="eval_report.json")
    args = ap.parse_args()

    examples = []
    with open(args.eval_file) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    if args.limit:
        examples = examples[: args.limit]

    print(f"Loaded {len(examples)} eval examples")
    model, tok = load_model(args.base_model, args.adapter)

    results = {
        "n": len(examples),
        "n_parsed": 0,
        "n_correct_relevance": 0,
        "pred_types": [],
        "gold_types": [],
        "per_example": [],
        "pred_summaries": [],
        "gold_summaries": [],
    }

    classes = VALID_TYPES + ["irrelevant"]

    for i, ex in enumerate(examples):
        gold_obj = json.loads(ex["output"])
        raw = generate(model, tok, ex["input"])
        pred_obj, ok = parse_output(raw)

        if ok:
            results["n_parsed"] += 1

        gold_rel = bool(gold_obj.get("relevant", False))
        pred_rel = bool(pred_obj.get("relevant", False)) if pred_obj else False
        if gold_rel == pred_rel:
            results["n_correct_relevance"] += 1

        gt = first_type(gold_obj)
        pt = first_type(pred_obj) if pred_obj else "irrelevant"
        results["pred_types"].append(pt)
        results["gold_types"].append(gt)

        gs = first_summary(gold_obj)
        ps = first_summary(pred_obj) if pred_obj else None
        if gs and ps:
            results["pred_summaries"].append(ps)
            results["gold_summaries"].append(gs)

        results["per_example"].append({
            "i": i,
            "input": ex["input"][:200],
            "gold": ex["output"][:300],
            "pred_raw": raw[:300],
            "parse_ok": ok,
            "rel_correct": gold_rel == pred_rel,
            "type_correct": gt == pt,
        })

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(examples)} done")

    # Aggregate
    per_class, macro = macro_f1(results["pred_types"], results["gold_types"], classes)
    summary = {
        "n_examples": results["n"],
        "json_parse_rate": results["n_parsed"] / max(results["n"], 1),
        "relevance_accuracy": results["n_correct_relevance"] / max(results["n"], 1),
        "type_macro_f1": macro,
        "type_per_class": per_class,
        "n_summary_pairs_for_bertscore": len(results["pred_summaries"]),
    }

    if args.bertscore and results["pred_summaries"]:
        try:
            from bert_score import score as bertscore
            P, R, F1 = bertscore(
                results["pred_summaries"],
                results["gold_summaries"],
                model_type="microsoft/deberta-xlarge-mnli",
                verbose=False,
            )
            summary["summary_bertscore_f1_mean"] = float(F1.mean())
        except Exception as e:
            summary["summary_bertscore_f1_mean"] = None
            summary["bertscore_error"] = str(e)

    Path(args.output).write_text(json.dumps(
        {"summary": summary, "per_example": results["per_example"]},
        indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

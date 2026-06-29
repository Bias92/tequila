"""
Overfit diagnostic for the C1 (agenda) QLoRA model.

Runs the fine-tuned checkpoint on a TRAIN sample and a VALID sample and prints
the metrics side-by-side. The TRAIN-minus-VALID GAP is the overfit verdict:
  - type/relevance gap large  -> model memorized train, doesn't generalize
  - summary_exact_match gap large -> verbatim memorization of train summaries

Fixes the dl[0]-only bug in eval_qlora.py: type scoring here uses the FULL
detail_list (set-based micro-F1 over all entries), so multi-entry utterances
(~8% of relevant) are scored, not just their first entry.

Usage (on RunPod):
  python overfit_diag.py \
    --adapter /workspace/checkpoints/atlas-llama32-3b-qlora \
    --train_file /workspace/atlas/data/challenge_data/train/train.jsonl \
    --valid_file /workspace/atlas/data/challenge_data/valid/valid.jsonl \
    --n 200
"""

import argparse
import json
import random
import re

from eval_qlora import load_model, generate, parse_output  # reuse exact infra


def type_set(parsed):
    """All types in the prediction's detail_list (not just the first)."""
    if not parsed or not parsed.get("relevant"):
        return set()
    return {d.get("type") for d in (parsed.get("detail_list") or []) if d.get("type")}


def summaries(parsed):
    if not parsed or not parsed.get("relevant"):
        return []
    return [(d.get("summary") or "").strip()
            for d in (parsed.get("detail_list") or []) if d.get("summary")]


def norm(s):
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def eval_split(model, tok, examples):
    n = len(examples)
    n_parsed = rel_ok = 0
    tp = fp = fn = 0                      # micro over ALL type entries
    exact = total_pred_sum = 0           # verbatim memorization
    for ex in examples:
        gold = json.loads(ex["output"])
        raw = generate(model, tok, ex["input"])
        pred, ok = parse_output(raw)
        if ok:
            n_parsed += 1
        gr = bool(gold.get("relevant"))
        pr = bool(pred.get("relevant")) if pred else False
        if gr == pr:
            rel_ok += 1
        gset, pset = type_set(gold), type_set(pred)
        tp += len(gset & pset)
        fp += len(pset - gset)
        fn += len(gset - pset)
        gsums = {norm(s) for s in summaries(gold)}
        for ps in summaries(pred):
            total_pred_sum += 1
            if norm(ps) in gsums:        # predicted summary == a gold summary verbatim
                exact += 1
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return {
        "n": n,
        "json_parse_rate": round(n_parsed / max(n, 1), 4),
        "relevance_acc": round(rel_ok / max(n, 1), 4),
        "type_micro_f1": round(f1, 4),
        "type_precision": round(prec, 4),
        "type_recall": round(rec, 4),
        "summary_exact_match_rate": round(exact / max(total_pred_sum, 1), 4),
        "n_pred_summaries": total_pred_sum,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="meta-llama/Llama-3.2-3B-Instruct")
    ap.add_argument("--adapter", default="/workspace/checkpoints/atlas-llama32-3b-qlora")
    ap.add_argument("--train_file", required=True)
    ap.add_argument("--valid_file", required=True)
    ap.add_argument("--n", type=int, default=200, help="samples per split")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)

    def load(path):
        rows = [json.loads(l) for l in open(path) if l.strip()]
        random.shuffle(rows)
        return rows[: args.n]

    tr, va = load(args.train_file), load(args.valid_file)
    model, tok = load_model(args.base_model, args.adapter)

    rtr = eval_split(model, tok, tr)
    rva = eval_split(model, tok, va)

    print("\n=== TRAIN sample ===")
    print(json.dumps(rtr, indent=2))
    print("\n=== VALID sample ===")
    print(json.dumps(rva, indent=2))
    print("\n=== GAP (train - valid) — 클수록 overfit ===")
    for k in ["json_parse_rate", "relevance_acc", "type_micro_f1", "summary_exact_match_rate"]:
        g = rtr[k] - rva[k]
        print(f"  {k:26s} train={rtr[k]:.3f}  valid={rva[k]:.3f}  gap={g:+.3f}")
    print("\n해석: type_micro_f1/relevance gap이 크면 task-level 일반화 실패(=overfit).")
    print("      summary_exact_match가 train만 높으면 = train summary 통째로 암기(verbatim memorization).")


if __name__ == "__main__":
    main()

"""AMI answer-linking baseline with an OpenAI LLM judge.

This evaluates the one AMI signal that is clean enough for cross-domain
checking: positive adjacency pairs. For each elicit act that has a human
annotated target, the model chooses which candidate utterance answered it.

The script reads OPENAI_API_KEY from the environment only.

Example:
  cd /Users/markov/Desktop/tequila/experiments/solution2_probe
  export OPENAI_API_KEY='<OPENAI_API_KEY>'
  python3 run_ami_llm_judge_openai.py --model gpt-4o --limit 200 \
    --out outputs/ami_llm_judge_gpt4o200.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

from run_edge_scorer_probe import ami_edges_for_row, load_ami_rows
from run_llm_judge_baseline import append_jsonl, clean_json, pct
from run_llm_judge_openai import openai_call


SYSTEM_PROMPT = """You judge answer links in meeting dialogue.

You will receive one question-like utterance and a list of later candidate
utterances from the same meeting. Choose the candidate that best answers the
question. If none of the candidates answers it, return null.

Return JSON only:
{
  "answered_by_act_id": string or null,
  "confidence": "high" | "medium" | "low",
  "rationale": short string
}
"""

SYSTEM_PROMPT_FORCE = """You judge answer links in meeting dialogue.

You will receive one question-like utterance and a list of later candidate
utterances from the same meeting. In this evaluation set, each question has
at least one human-annotated answer among the candidate utterances. Choose the
candidate that best answers the question.

Return JSON only:
{
  "answered_by_act_id": string,
  "confidence": "high" | "medium" | "low",
  "rationale": short string
}
"""


def row_key(row: tuple[Any, ...]) -> str:
    meeting, _acts, q_idx, _targets, _summaries = row
    return f"{meeting}:{_acts[q_idx].act_id}"


def gold_distance(row: tuple[Any, ...]) -> int | None:
    _meeting, acts, q_idx, targets, _summaries = row
    index_by_id = {act.act_id: idx for idx, act in enumerate(acts)}
    distances = [
        index_by_id[target] - q_idx
        for target in targets
        if target in index_by_id and index_by_id[target] > q_idx
    ]
    return min(distances) if distances else None


def distance_bucket(distance: int | None) -> str:
    if distance is None:
        return "unknown"
    if distance == 1:
        return "d1"
    if distance <= 3:
        return "d2_3"
    return "d4p"


def make_user_prompt(
    row: tuple[Any, ...], max_candidates: int, force_choice: bool
) -> tuple[str, list[dict[str, Any]]]:
    meeting, acts, q_idx, _targets, summaries = row
    question = acts[q_idx]
    edges = ami_edges_for_row(row)
    candidates: list[dict[str, Any]] = []
    for edge in edges[:max_candidates]:
        candidates.append(
            {
                "act_id": edge.cand_key[1],
                "speaker": next(
                    (act.speaker for act in acts if act.act_id == edge.cand_key[1]),
                    "",
                ),
                "da_type": edge.cand_type,
                "distance": edge.distance,
                "summary": edge.cand_summary,
                "text": edge.cand_text,
            }
        )

    prompt = {
        "meeting": meeting,
        "question": {
            "act_id": question.act_id,
            "speaker": question.speaker,
            "da_type": question.da_type,
            "summary": summaries.get((meeting, question.act_id), question.summary),
            "text": question.text,
        },
        "candidates": candidates,
        "instruction": (
            "Pick exactly one candidate act_id. Do not return null."
            if force_choice
            else "Pick exactly one candidate act_id if it answers the question. "
            "Return null if none of the candidates answers it."
        ),
    }
    return json.dumps(prompt, ensure_ascii=False, indent=2), candidates


def load_done(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    out: dict[str, dict[str, Any]] = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            out[item["key"]] = item
    return out


def normalize_prediction(pred: dict[str, Any]) -> str | None:
    value = pred.get("answered_by_act_id")
    if value is None:
        return None
    value = str(value).strip()
    if value.lower() in {"", "null", "none", "no_answer", "no answer"}:
        return None
    return value


def compute_metrics(items: list[dict[str, Any]]) -> dict[str, Any]:
    total = 0
    hit = 0
    none = 0
    by_bucket: dict[str, dict[str, int]] = {
        "d1": {"hit": 0, "total": 0},
        "d2_3": {"hit": 0, "total": 0},
        "d4p": {"hit": 0, "total": 0},
        "unknown": {"hit": 0, "total": 0},
    }
    for item in items:
        if "prediction" not in item:
            continue
        total += 1
        pred = normalize_prediction(item["prediction"])
        if pred is None:
            none += 1
        ok = pred in set(item["gold_targets"])
        if ok:
            hit += 1
        bucket = item.get("distance_bucket", "unknown")
        by_bucket.setdefault(bucket, {"hit": 0, "total": 0})
        by_bucket[bucket]["total"] += 1
        by_bucket[bucket]["hit"] += int(ok)
    return {"total": total, "hit": hit, "none": none, "by_bucket": by_bucket}


def print_metrics(metrics: dict[str, Any]) -> None:
    total = metrics["total"]
    print(json.dumps(metrics, indent=2))
    print("link_accuracy", pct(metrics["hit"], total))
    print("none_rate", pct(metrics["none"], total))
    for bucket, vals in metrics["by_bucket"].items():
        if vals["total"]:
            print(f"{bucket}_accuracy", pct(vals["hit"], vals["total"]))


def select_rows(rows: list[tuple[Any, ...]], limit: int | None, seed: int) -> list[tuple[Any, ...]]:
    rows = sorted(rows, key=row_key)
    if limit is None or limit >= len(rows):
        return rows
    rng = random.Random(seed)
    selected = rng.sample(rows, limit)
    return sorted(selected, key=row_key)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--meetings", type=int, default=5)
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--max-candidates", type=int, default=20)
    parser.add_argument("--out", type=Path, default=Path("outputs/ami_llm_judge_openai.jsonl"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--score-only", action="store_true")
    parser.add_argument("--force-choice", action="store_true")
    args = parser.parse_args()

    rows = load_ami_rows(meeting_limit=args.meetings, window=args.window)
    rows = select_rows(rows, args.limit, args.seed)
    print(
        f"ami_answer_link_rows={len(rows)} meetings={args.meetings} "
        f"window={args.window} model={args.model}"
    )

    if args.dry_run:
        if not rows:
            return 0
        prompt, _candidates = make_user_prompt(rows[0], args.max_candidates, args.force_choice)
        print(prompt)
        return 0

    done = load_done(args.out)
    if args.score_only:
        print_metrics(compute_metrics(list(done.values())))
        return 0

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set in this shell")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    for idx, row in enumerate(rows, 1):
        key = row_key(row)
        if key in done:
            continue
        prompt, candidates = make_user_prompt(row, args.max_candidates, args.force_choice)
        raw = openai_call(
            args.model,
            SYSTEM_PROMPT_FORCE if args.force_choice else SYSTEM_PROMPT,
            prompt,
            args.temperature,
            args.retries,
        )
        pred = clean_json(raw)
        item = {
            "key": key,
            "meeting": row[0],
            "question_id": row[1][row[2]].act_id,
            "gold_targets": sorted(row[3]),
            "gold_distance": gold_distance(row),
            "distance_bucket": distance_bucket(gold_distance(row)),
            "candidate_ids": [c["act_id"] for c in candidates],
            "prediction": pred,
            "raw": raw,
        }
        append_jsonl(args.out, item)
        done[key] = item
        if args.sleep:
            time.sleep(args.sleep)
        if idx % 10 == 0:
            print(f"processed {idx}/{len(rows)}")

    print_metrics(compute_metrics(list(done.values())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

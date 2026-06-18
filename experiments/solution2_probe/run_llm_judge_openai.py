"""LLM-judge baseline using OpenAI instead of Gemini.

Reuses the EXACT same data (real ACI via iter_question_rows), prompt, row
selection, and metrics as run_llm_judge_baseline.py — only the API call is
swapped to OpenAI. Reads OPENAI_API_KEY from the environment only.

Usage (key in your shell env, NOT in the command):
  export OPENAI_API_KEY='<OPENAI_API_KEY>'
  python3 run_llm_judge_openai.py --limit 100 --model gpt-4o --out outputs/llm_judge_gpt4o100.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import ssl
import time
import urllib.error
import urllib.request
from pathlib import Path

from run_broad_dual_probe import corpora, iter_question_rows
from run_llm_judge_baseline import (
    SYSTEM_PROMPT,
    append_jsonl,
    clean_json,
    compute_metrics,
    load_done,
    make_user_prompt,
    pct,
    row_key,
    select_rows,
)


def openai_call(model: str, system: str, user: str, temperature: float, retries: int) -> str:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    url = "https://api.openai.com/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
        "response_format": {"type": "json_object"},
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + key},
    )
    try:
        import certifi

        ctx = ssl.create_default_context(cafile=certifi.where())
    except Exception:
        ctx = ssl.create_default_context()

    last_error = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=90, context=ctx) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"]
        except (urllib.error.HTTPError, urllib.error.URLError, KeyError, TimeoutError) as exc:
            last_error = exc
            sleep = 2 ** attempt
            if isinstance(exc, urllib.error.HTTPError) and exc.code in {429, 500, 502, 503, 504}:
                sleep = max(sleep, 20)
            time.sleep(sleep)
    raise RuntimeError(f"OpenAI call failed after {retries} retries: {last_error}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="all_json_annotations", choices=sorted(corpora()))
    parser.add_argument("--speaker", default="patient")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--out", type=Path, default=Path("outputs/llm_judge_openai.jsonl"))
    parser.add_argument("--no-counterfactual", action="store_true")
    args = parser.parse_args()

    rows = list(iter_question_rows(corpora()[args.corpus], speaker_filter=args.speaker))
    rows = select_rows(rows, args.limit, args.seed)
    rows.sort(key=lambda r: (str(r["path"]), r["question_utterance"]["id"]))
    print(
        f"corpus={args.corpus} rows={len(rows)} "
        f"answerable={sum(r['silver_answer'] is not None for r in rows)} "
        f"no_silver={sum(r['silver_answer'] is None for r in rows)}"
    )

    done = load_done(args.out)
    for idx, row in enumerate(rows, 1):
        modes = ["actual"]
        if row["silver_answer"] is not None and not args.no_counterfactual:
            modes.append("counterfactual")
        for mode in modes:
            key = row_key(row, mode)
            if key in done:
                continue
            user_prompt = make_user_prompt(row, counterfactual=(mode == "counterfactual"))
            raw = openai_call(args.model, SYSTEM_PROMPT, user_prompt, args.temperature, args.retries)
            pred = clean_json(raw)
            item = {
                "key": key,
                "mode": mode,
                "path": str(row["path"]),
                "question_id": row["question_utterance"]["id"],
                "silver_answer_id": (
                    row["silver_answer"]["utterance"]["id"]
                    if row["silver_answer"] is not None
                    else None
                ),
                "prediction": pred,
                "raw": raw,
            }
            append_jsonl(args.out, item)
            done[key] = item
            if args.sleep:
                time.sleep(args.sleep)
        if idx % 10 == 0:
            print(f"processed {idx}/{len(rows)}")

    metrics = compute_metrics(list(done.values()))
    print(json.dumps(metrics, indent=2))
    print("answered_status", pct(metrics["answered_status"], metrics["answerable"]))
    print("answered_exact", pct(metrics["answered_exact"], metrics["answerable"]))
    print("no_silver_unanswered", pct(metrics["no_silver_unanswered"], metrics["no_silver"]))
    print("counterfactual_unanswered", pct(metrics["counterfactual_unanswered"], metrics["counterfactual_n"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

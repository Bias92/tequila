from __future__ import annotations

import argparse
import json
import os
import random
import re
import ssl
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from run_broad_dual_probe import (
    LOOKAHEAD_UTTERANCES,
    corpora,
    iter_question_rows,
)


SYSTEM_PROMPT = """You are judging whether a patient's clinical question was answered later in the dialogue.

Return only valid JSON with this schema:
{"answered": true/false, "answered_by_utterance_id": string|null, "rationale": string}

Rules:
- Mark answered=true only if a later utterance directly answers the patient's question.
- If the later utterances are unrelated, only acknowledge the patient, or do not address the question, mark answered=false.
- Use only the provided utterances. Do not infer from medical knowledge.
- If answered=true, answered_by_utterance_id must be the first utterance that answers the question.
"""


def clean_json(text: str) -> dict | None:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.S)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None


def make_user_prompt(row: dict, counterfactual: bool = False) -> str:
    q_utt = row["question_utterance"]
    q_idx = row["question_index"]
    utterances = row["utterances"]
    end = min(len(utterances), q_idx + LOOKAHEAD_UTTERANCES + 1)
    skip_id = None
    if counterfactual and row["silver_answer"] is not None:
        skip_id = row["silver_answer"]["utterance"]["id"]

    following = []
    for utt in utterances[q_idx + 1 : end]:
        if utt["id"] == skip_id:
            continue
        following.append(
            f'- {utt["id"]} [{utt.get("speaker", "")}]: {utt.get("text", "")}'
        )
    following_text = "\n".join(following) if following else "(none)"

    return f"""Patient question:
- {q_utt["id"]} [{q_utt.get("speaker", "")}]: {q_utt.get("text", "")}

Question summary:
{row["question_detail"].get("summary", "")}

Later utterances within the next {LOOKAHEAD_UTTERANCES} turns:
{following_text}

Was the patient's question answered by one of the later utterances?"""


def gemini_call(model: str, system: str, user: str, temperature: float, retries: int) -> str:
    key = os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("GOOGLE_API_KEY is not set")

    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        + model
        + ":generateContent?key="
        + key
    )
    payload = {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": user}]}],
        "generationConfig": {
            "temperature": temperature,
            "responseMimeType": "application/json",
        },
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    last_error = None
    try:
        import certifi

        ssl_context = ssl.create_default_context(cafile=certifi.where())
    except Exception:
        ssl_context = ssl.create_default_context()

    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=90, context=ssl_context) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except (urllib.error.HTTPError, urllib.error.URLError, KeyError, TimeoutError) as exc:
            last_error = exc
            sleep = 2 ** attempt
            if isinstance(exc, urllib.error.HTTPError) and exc.code in {429, 500, 502, 503, 504}:
                sleep = max(sleep, 20)
            time.sleep(sleep)
    raise RuntimeError(f"Gemini call failed after {retries} retries: {last_error}")


def row_key(row: dict, mode: str) -> str:
    return f"{row['path']}::{row['question_utterance']['id']}::{mode}"


def load_done(path: Path) -> dict:
    done = {}
    if not path.exists():
        return done
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        done[item["key"]] = item
    return done


def append_jsonl(path: Path, item: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


def select_rows(rows: list[dict], limit: int | None, seed: int) -> list[dict]:
    if limit is None or limit >= len(rows):
        return rows
    rng = random.Random(seed)
    answerable = [row for row in rows if row["silver_answer"] is not None]
    no_silver = [row for row in rows if row["silver_answer"] is None]
    # Keep a visible no-answer slice even in small pilots.
    no_n = min(len(no_silver), limit, max(1, round(limit * 0.15)))
    ans_n = max(0, limit - no_n)
    return rng.sample(answerable, min(ans_n, len(answerable))) + rng.sample(
        no_silver, min(no_n, len(no_silver))
    )


def compute_metrics(items: list[dict]) -> dict:
    actual = [x for x in items if x["mode"] == "actual"]
    cf = [x for x in items if x["mode"] == "counterfactual"]

    answerable_actual = [x for x in actual if x["silver_answer_id"] is not None]
    no_silver_actual = [x for x in actual if x["silver_answer_id"] is None]

    def is_answered(x):
        return bool(x.get("prediction", {}).get("answered"))

    def pred_id(x):
        return x.get("prediction", {}).get("answered_by_utterance_id")

    return {
        "actual_n": len(actual),
        "answerable": len(answerable_actual),
        "no_silver": len(no_silver_actual),
        "answered_status": sum(is_answered(x) for x in answerable_actual),
        "answered_exact": sum(
            is_answered(x) and pred_id(x) == x["silver_answer_id"]
            for x in answerable_actual
        ),
        "no_silver_unanswered": sum(not is_answered(x) for x in no_silver_actual),
        "counterfactual_n": len(cf),
        "counterfactual_unanswered": sum(not is_answered(x) for x in cf),
    }


def pct(k: int, n: int) -> str:
    return "NA" if n == 0 else f"{k}/{n} ({k / n:.1%})"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="all_json_annotations", choices=sorted(corpora()))
    parser.add_argument("--speaker", default="patient")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--model", default="gemini-2.5-flash")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--out", type=Path, default=Path("outputs/llm_judge_baseline.jsonl"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-counterfactual", action="store_true")
    args = parser.parse_args()

    corpus_map = corpora()
    rows = list(iter_question_rows(corpus_map[args.corpus], speaker_filter=args.speaker))
    rows = select_rows(rows, args.limit, args.seed)
    rows.sort(key=lambda r: (str(r["path"]), r["question_utterance"]["id"]))

    print(
        f"corpus={args.corpus} rows={len(rows)} "
        f"answerable={sum(r['silver_answer'] is not None for r in rows)} "
        f"no_silver={sum(r['silver_answer'] is None for r in rows)}"
    )

    if args.dry_run:
        row = rows[0]
        print("SYSTEM:")
        print(SYSTEM_PROMPT)
        print("USER:")
        print(make_user_prompt(row, counterfactual=False))
        return 0

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
            raw = gemini_call(
                args.model,
                SYSTEM_PROMPT,
                user_prompt,
                args.temperature,
                args.retries,
            )
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

    items = list(done.values())
    metrics = compute_metrics(items)
    print(json.dumps(metrics, indent=2))
    print("answered_status", pct(metrics["answered_status"], metrics["answerable"]))
    print("answered_exact", pct(metrics["answered_exact"], metrics["answerable"]))
    print("no_silver_unanswered", pct(metrics["no_silver_unanswered"], metrics["no_silver"]))
    print(
        "counterfactual_unanswered",
        pct(metrics["counterfactual_unanswered"], metrics["counterfactual_n"]),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

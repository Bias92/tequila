from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path

from src.state_tracker_solution2 import ANSWER_CANDIDATE_TYPES, Solution2Tracker
from src.state_tracker_solution2_hybrid import HybridSolution2Tracker


DEFAULT_ATLAS_DATA = Path(__file__).resolve().parents[3] / "atlas" / "data" / "challenge_data"
ATLAS_DATA = Path(os.environ.get("ATLAS_DATA", DEFAULT_ATLAS_DATA))
SPLITS = [
    "train",
    "valid",
    "clinicalnlp_taskB_test1",
    "clinicalnlp_taskC_test2",
    "clef_taskC_test3",
]
LOOKAHEAD_UTTERANCES = 4


def clean_detail(detail):
    return {"type": detail["type"], "summary": detail["summary"]}


def iter_patient_questions():
    for split in SPLITS:
        aci_dir = ATLAS_DATA / split / "aci"
        for path in sorted(aci_dir.glob("*.json")):
            data = json.loads(path.read_text())
            utterances = data["utterances"]
            ids = [u["id"] for u in utterances]
            utt_by_id = {u["id"]: u for u in utterances}
            ann_by_id = {a["utterance_id"]: a for a in data.get("annotations", [])}

            for ann in data.get("annotations", []):
                utt = utt_by_id[ann["utterance_id"]]
                if utt.get("speaker") != "patient":
                    continue

                for detail in ann.get("detail_list") or []:
                    if detail.get("type") != "question":
                        continue

                    q_idx = ids.index(ann["utterance_id"])
                    silver_answer = find_silver_answer(
                        q_idx,
                        utterances,
                        ids,
                        utt_by_id,
                        ann_by_id,
                    )
                    yield {
                        "split": split,
                        "file": path.name,
                        "utterances": utterances,
                        "ids": ids,
                        "ann_by_id": ann_by_id,
                        "question_index": q_idx,
                        "question_utterance": utt,
                        "question_detail": detail,
                        "silver_answer": silver_answer,
                    }


def find_silver_answer(q_idx, utterances, ids, utt_by_id, ann_by_id):
    q_utt = utterances[q_idx]
    for offset in range(1, LOOKAHEAD_UTTERANCES + 1):
        idx = q_idx + offset
        if idx >= len(utterances):
            break
        utt = utt_by_id[ids[idx]]
        if utt["speaker"] == q_utt["speaker"]:
            continue

        ann = ann_by_id.get(utt["id"])
        if ann:
            answer_detail = next(
                (
                    d
                    for d in ann.get("detail_list") or []
                    if d.get("type") in ANSWER_CANDIDATE_TYPES
                ),
                None,
            )
            if answer_detail:
                return {
                    "utterance": utt,
                    "detail": answer_detail,
                    "kind": "annotated_summary",
                    "offset": offset,
                }

        if is_raw_short_answer(utt["text"]):
            return {
                "utterance": utt,
                "detail": None,
                "kind": "raw_short_answer",
                "offset": offset,
            }
    return None


def is_raw_short_answer(text: str) -> bool:
    cleaned = []
    for ch in text.lower():
        cleaned.append(ch if ch.isalnum() or ch == "'" else " ")
    words = "".join(cleaned).split()
    if not words or len(words) > 8:
        return False
    markers = {
        "yes", "yeah", "yep", "correct", "right", "exactly", "absolutely",
        "no", "nope", "not", "dont", "don't", "doesnt", "doesn't",
    }
    ack_only = {"okay", "ok", "right", "mm", "mmhmm", "mhm", "uh", "huh"}
    return not set(words) <= ack_only and bool(set(words) & markers)


def run_window(row, tracker_cls, include_answer=True):
    tracker = tracker_cls()
    q_idx = row["question_index"]
    start = q_idx
    end = min(len(row["utterances"]), q_idx + LOOKAHEAD_UTTERANCES + 1)
    skip_answer_id = None
    if not include_answer and row["silver_answer"] is not None:
        skip_answer_id = row["silver_answer"]["utterance"]["id"]

    for utt in row["utterances"][start:end]:
        if utt["id"] == skip_answer_id:
            continue
        ann = row["ann_by_id"].get(utt["id"])
        if utt["id"] == row["question_utterance"]["id"]:
            ann = {
                "utterance_id": utt["id"],
                "relevant": True,
                "detail_list": [clean_detail(row["question_detail"])],
            }
        elif ann:
            ann = {
                "utterance_id": utt["id"],
                "relevant": ann["relevant"],
                "detail_list": [clean_detail(d) for d in ann.get("detail_list") or []],
            }
        tracker.update(utt, ann)

    tracker.finalize()
    return next(
        q
        for q in tracker.report()
        if q["utterance_id"] == row["question_utterance"]["id"]
    )


def evaluate(tracker_cls, rows):
    answerable = [row for row in rows if row["silver_answer"] is not None]
    actuals = [(row, run_window(row, tracker_cls, include_answer=True)) for row in answerable]
    counterfactuals = [
        (row, run_window(row, tracker_cls, include_answer=False))
        for row in answerable
    ]

    exact = [
        (row, actual)
        for row, actual in actuals
        if (
            actual["status"] == "answered"
            and actual["answered_by_utterance_id"]
            == row["silver_answer"]["utterance"]["id"]
        )
    ]
    status = [(row, actual) for row, actual in actuals if actual["status"] == "answered"]
    unanswered = [
        (row, actual)
        for row, actual in counterfactuals
        if actual["status"] == "unanswered"
    ]
    return {
        "answerable": answerable,
        "actuals": actuals,
        "exact": exact,
        "status": status,
        "counterfactuals": counterfactuals,
        "unanswered": unanswered,
    }


def print_eval(name, result):
    total = len(result["answerable"])
    exact = len(result["exact"])
    status = len(result["status"])
    unanswered = len(result["unanswered"])
    print(f"\n{name}")
    print(f"answered exact: {exact}/{total} ({exact / max(1, total):.1%})")
    print(f"answered status-only: {status}/{total} ({status / max(1, total):.1%})")
    print(
        "counterfactual unanswered: "
        f"{unanswered}/{len(result['counterfactuals'])} "
        f"({unanswered / max(1, len(result['counterfactuals'])):.1%})"
    )
    by_split_total = Counter(row["split"] for row in result["answerable"])
    by_split_exact = Counter(row["split"] for row, _ in result["exact"])
    for split in SPLITS:
        if by_split_total[split]:
            print(
                f"  {split}: {by_split_exact[split]}/{by_split_total[split]} "
                f"({by_split_exact[split] / by_split_total[split]:.1%})"
            )


def print_failures(name, result, limit=12):
    failures = [
        (row, actual)
        for row, actual in result["actuals"]
        if actual["status"] != "answered"
        or actual["answered_by_utterance_id"] != row["silver_answer"]["utterance"]["id"]
    ]
    print(f"\n{name} FAILURES, FIRST {limit}")
    for row, actual in failures[:limit]:
        answer = row["silver_answer"]
        print(
            f"- {row['split']}/{row['file']} "
            f"{row['question_utterance']['id']} -> {answer['utterance']['id']} "
            f"kind={answer['kind']} actual={actual['status']}/"
            f"{actual['answered_by_utterance_id']} "
            f"score={actual['match_score']} terms={actual['overlap_terms']}"
        )
        print(f"  Q: {row['question_detail']['summary']}")
        if answer["detail"]:
            print(f"  A: {answer['detail']['summary']}")
        else:
            print(f"  raw A: {answer['utterance']['text']}")


def main():
    rows = list(iter_patient_questions())
    answerable = [row for row in rows if row["silver_answer"] is not None]
    raw_short = [row for row in answerable if row["silver_answer"]["kind"] == "raw_short_answer"]
    annotated = [row for row in answerable if row["silver_answer"]["kind"] == "annotated_summary"]

    print("DATASET SCAN")
    print(f"patient questions: {len(rows)}")
    print(f"silver answerable within {LOOKAHEAD_UTTERANCES} utterances: {len(answerable)}")
    print(f"  annotated summary answers: {len(annotated)}")
    print(f"  raw short answers: {len(raw_short)}")
    print(f"no silver answer candidate: {len(rows) - len(answerable)}")

    lexical = evaluate(Solution2Tracker, rows)
    hybrid = evaluate(HybridSolution2Tracker, rows)

    print_eval("LEXICAL BASELINE", lexical)
    print_eval("HYBRID RAW+EMBEDDING", hybrid)
    print_failures("HYBRID RAW+EMBEDDING", hybrid)


if __name__ == "__main__":
    main()

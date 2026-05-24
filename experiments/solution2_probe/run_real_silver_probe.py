import json
import os
from collections import Counter
from pathlib import Path

from src.state_tracker_solution2 import ANSWER_CANDIDATE_TYPES, Solution2Tracker


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
    return {
        "type": detail["type"],
        "summary": detail["summary"],
    }


def iter_patient_question_candidates():
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

                    question_index = ids.index(ann["utterance_id"])
                    answer = None
                    for offset in range(1, LOOKAHEAD_UTTERANCES + 1):
                        next_index = question_index + offset
                        if next_index >= len(ids):
                            break
                        next_utt = utt_by_id[ids[next_index]]
                        next_ann = ann_by_id.get(next_utt["id"])
                        if not next_ann:
                            continue
                        if next_utt.get("speaker") == utt.get("speaker"):
                            continue

                        answer_detail = next(
                            (
                                d
                                for d in next_ann.get("detail_list") or []
                                if d.get("type") in ANSWER_CANDIDATE_TYPES
                            ),
                            None,
                        )
                        if answer_detail:
                            answer = {
                                "utterance": next_utt,
                                "detail": answer_detail,
                                "offset": offset,
                            }
                            break

                    yield {
                        "split": split,
                        "file": path.name,
                        "question_utterance": utt,
                        "question_detail": detail,
                        "answer": answer,
                    }


def run_minicase(question_utt, question_detail, answer=None):
    tracker = Solution2Tracker()

    tracker.update(
        question_utt,
        {
            "utterance_id": question_utt["id"],
            "relevant": True,
            "detail_list": [clean_detail(question_detail)],
        },
    )

    if answer is not None:
        tracker.update(
            answer["utterance"],
            {
                "utterance_id": answer["utterance"]["id"],
                "relevant": True,
                "detail_list": [clean_detail(answer["detail"])],
            },
        )

    tracker.finalize()
    report = tracker.report()
    return next(q for q in report if q["utterance_id"] == question_utt["id"])


def main() -> int:
    rows = list(iter_patient_question_candidates())
    answered_rows = [r for r in rows if r["answer"] is not None]
    no_candidate_rows = [r for r in rows if r["answer"] is None]

    print("DATASET SCAN")
    print(f"patient questions: {len(rows)}")
    print(
        f"silver answered candidates within {LOOKAHEAD_UTTERANCES} utterances: "
        f"{len(answered_rows)}"
    )
    print(f"no near answer candidate: {len(no_candidate_rows)}")

    answered_results = []
    counterfactual_results = []
    for row in answered_rows:
        actual = run_minicase(
            row["question_utterance"],
            row["question_detail"],
            row["answer"],
        )
        answered_results.append((row, actual))

        counterfactual = run_minicase(
            row["question_utterance"],
            row["question_detail"],
            answer=None,
        )
        counterfactual_results.append((row, counterfactual))

    answered_exact = [
        (row, actual)
        for row, actual in answered_results
        if (
            actual["status"] == "answered"
            and actual["answered_by_utterance_id"] == row["answer"]["utterance"]["id"]
        )
    ]
    answered_status_only = [
        (row, actual)
        for row, actual in answered_results
        if actual["status"] == "answered"
    ]
    unanswered_ok = [
        (row, actual)
        for row, actual in counterfactual_results
        if actual["status"] == "unanswered"
    ]

    print("\nRESULTS")
    print(
        "answered exact: "
        f"{len(answered_exact)}/{len(answered_results)} "
        f"({len(answered_exact) / max(1, len(answered_results)):.1%})"
    )
    print(
        "answered status-only: "
        f"{len(answered_status_only)}/{len(answered_results)} "
        f"({len(answered_status_only) / max(1, len(answered_results)):.1%})"
    )
    print(
        "counterfactual unanswered: "
        f"{len(unanswered_ok)}/{len(counterfactual_results)} "
        f"({len(unanswered_ok) / max(1, len(counterfactual_results)):.1%})"
    )

    by_split = Counter(row["split"] for row, _ in answered_exact)
    total_by_split = Counter(row["split"] for row in answered_rows)
    print("\nANSWERED EXACT BY SPLIT")
    for split in SPLITS:
        total = total_by_split[split]
        correct = by_split[split]
        if total:
            print(f"{split}: {correct}/{total} ({correct / total:.1%})")

    failures = [
        (row, actual)
        for row, actual in answered_results
        if actual["status"] != "answered"
    ]
    print("\nANSWERED FAILURES, FIRST 12")
    for row, actual in failures[:12]:
        answer = row["answer"]
        print(
            f"- {row['split']}/{row['file']} "
            f"{row['question_utterance']['id']} -> "
            f"{answer['utterance']['id']} "
            f"actual={actual['status']} score={actual['match_score']} "
            f"overlap={actual['overlap_terms']}"
        )
        print(f"  Q: {row['question_detail']['summary']}")
        print(f"  A: {answer['detail']['summary']}")

    print("\nNO-NEAR-CANDIDATE EXAMPLES, FIRST 8")
    for row in no_candidate_rows[:8]:
        print(f"- {row['split']}/{row['file']} {row['question_utterance']['id']}")
        print(f"  Q: {row['question_detail']['summary']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

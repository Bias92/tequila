import json
from pathlib import Path

from src.state_tracker_solution2 import Solution2Tracker


ROOT = Path(__file__).resolve().parent
CASE_DIR = ROOT / "data" / "sample_cases"


def run_case(path: Path) -> bool:
    case = json.loads(path.read_text())
    annotations = {a["utterance_id"]: a for a in case.get("annotations", [])}

    tracker = Solution2Tracker()
    for utt in case["utterances"]:
        tracker.update(utt, annotations.get(utt["id"]))
    tracker.finalize()

    report = tracker.report()
    by_utt = {q["utterance_id"]: q for q in report}
    ok = True

    print(f"\n=== {case['id']} ===")
    print(f"source: {case['source']}")

    for expected in case.get("expected_questions", []):
        actual = by_utt.get(expected["utterance_id"])
        if not actual:
            ok = False
            print(f"MISS question {expected['utterance_id']}: tracker produced nothing")
            continue

        status_ok = actual["status"] == expected["expected_status"]
        answer_ok = (
            actual["answered_by_utterance_id"]
            == expected["expected_answer_utterance_id"]
        )
        row_ok = status_ok and answer_ok
        ok = ok and row_ok

        verdict = "PASS" if row_ok else "FAIL"
        print(
            f"{verdict} {actual['utterance_id']}: "
            f"expected={expected['expected_status']}/"
            f"{expected['expected_answer_utterance_id']} "
            f"actual={actual['status']}/"
            f"{actual['answered_by_utterance_id']} "
            f"score={actual['match_score']} "
            f"overlap={actual['overlap_terms']}"
        )
        print(f"  Q: {actual['summary']}")
        if actual["answered_by_summary"]:
            print(f"  A: {actual['answered_by_summary']}")

    if not case.get("expected_questions"):
        print("No expected questions listed.")

    return ok


def main() -> int:
    paths = sorted(CASE_DIR.glob("*.json"))
    if not paths:
        print(f"No cases found in {CASE_DIR}")
        return 2

    results = [run_case(path) for path in paths]
    passed = sum(1 for r in results if r)
    print(f"\nSUMMARY: {passed}/{len(results)} cases passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())

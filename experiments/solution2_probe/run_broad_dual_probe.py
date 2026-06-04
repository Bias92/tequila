from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path

from src.state_tracker_solution2 import ANSWER_CANDIDATE_TYPES, Solution2Tracker
from src.state_tracker_solution2_hybrid import HybridSolution2Tracker


DEFAULT_ATLAS_ROOT = Path(__file__).resolve().parents[3] / "atlas"
ATLAS_ROOT = Path(os.environ.get("ATLAS_ROOT", DEFAULT_ATLAS_ROOT))
CHALLENGE = ATLAS_ROOT / "data" / "challenge_data"
SRC_EXPERIMENT = ATLAS_ROOT / "data" / "src_experiment_data"

SPLITS = [
    "train",
    "valid",
    "clinicalnlp_taskB_test1",
    "clinicalnlp_taskC_test2",
    "clef_taskC_test3",
]
LOOKAHEAD_UTTERANCES = 4


class CachedHybridTracker(HybridSolution2Tracker):
    """Hybrid tracker with a process-wide embedding cache for broad probes."""

    _global_embedding_cache = {}

    def _encode(self, text: str):
        cache = CachedHybridTracker._global_embedding_cache
        if text not in cache:
            cache[text] = self._get_encoder().encode(text, normalize_embeddings=True)
        return cache[text]


class CachedNoRawHybridTracker(CachedHybridTracker):
    """Hybrid tracker variant that disables raw short-answer closing."""

    def _try_raw_short_answer(self, utterance):
        return None


def clean_detail(detail):
    return {"type": detail["type"], "summary": detail["summary"]}


def is_raw_short_answer(text: str) -> bool:
    cleaned = []
    for ch in text.lower():
        cleaned.append(ch if ch.isalnum() or ch == "'" else " ")
    words = "".join(cleaned).split()
    if not words or len(words) > 8:
        return False
    markers = {
        "yes", "yeah", "yep", "correct", "right", "exactly", "absolutely",
        "sure", "no", "nope", "not", "dont", "don't", "doesnt",
        "doesn't", "cant", "can't", "will", "can", "should",
    }
    ack_only = {"okay", "ok", "right", "mm", "mmhmm", "mhm", "uh", "huh"}
    return not set(words) <= ack_only and bool(set(words) & markers)


def challenge_paths(kind: str):
    for split in SPLITS:
        for path in sorted((CHALLENGE / split / kind).glob("*.json")):
            yield path


def src_experiment_paths():
    for path in sorted(SRC_EXPERIMENT.glob("*/aci/*.json")):
        yield path


def corpora():
    challenge_aci = list(challenge_paths("aci"))
    challenge_eval = list(challenge_paths("eval"))
    src_experiment_aci = list(src_experiment_paths())
    return {
        "challenge_aci": challenge_aci,
        "challenge_eval": challenge_eval,
        "src_experiment_aci": src_experiment_aci,
        "all_json_annotations": challenge_aci + challenge_eval + src_experiment_aci,
    }


def load_docs(paths):
    for path in paths:
        data = json.loads(path.read_text())
        if isinstance(data.get("utterances"), list) and isinstance(data.get("annotations"), list):
            yield path, data


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
            for detail in ann.get("detail_list") or []:
                if detail.get("type") in ANSWER_CANDIDATE_TYPES:
                    return {
                        "utterance": utt,
                        "detail": detail,
                        "kind": "annotated_summary",
                        "offset": offset,
                    }

        if is_raw_short_answer(utt.get("text", "")):
            return {
                "utterance": utt,
                "detail": None,
                "kind": "raw_short_answer",
                "offset": offset,
            }
    return None


def iter_question_rows(paths, speaker_filter=None):
    for path, data in load_docs(paths):
        utterances = data["utterances"]
        ids = [utt["id"] for utt in utterances]
        utt_by_id = {utt["id"]: utt for utt in utterances}
        ann_by_id = {ann["utterance_id"]: ann for ann in data.get("annotations", [])}

        for ann in data.get("annotations", []):
            utt = utt_by_id.get(ann["utterance_id"])
            if not utt:
                continue
            if speaker_filter and utt.get("speaker") != speaker_filter:
                continue

            for detail in ann.get("detail_list") or []:
                if detail.get("type") != "question":
                    continue

                q_idx = ids.index(ann["utterance_id"])
                yield {
                    "path": path,
                    "dataset": path.parent.parent.name,
                    "speaker": utt.get("speaker"),
                    "utterances": utterances,
                    "ann_by_id": ann_by_id,
                    "question_index": q_idx,
                    "question_utterance": utt,
                    "question_detail": detail,
                    "silver_answer": find_silver_answer(
                        q_idx,
                        utterances,
                        ids,
                        utt_by_id,
                        ann_by_id,
                    ),
                }


def tracker_with_threshold(base_cls, embedding_threshold):
    class Tracker(base_cls):
        def __init__(self):
            super().__init__(embedding_threshold=embedding_threshold)

    return Tracker


def run_window(row, tracker_cls, include_answer=True):
    tracker = tracker_cls()
    q_idx = row["question_index"]
    end = min(len(row["utterances"]), q_idx + LOOKAHEAD_UTTERANCES + 1)
    skip_answer_id = None
    if not include_answer and row["silver_answer"] is not None:
        skip_answer_id = row["silver_answer"]["utterance"]["id"]

    for utt in row["utterances"][q_idx:end]:
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
        question
        for question in tracker.report()
        if question["utterance_id"] == row["question_utterance"]["id"]
    )


def evaluate(rows, tracker_cls):
    answerable = [row for row in rows if row["silver_answer"] is not None]
    no_silver = [row for row in rows if row["silver_answer"] is None]

    actuals = [(row, run_window(row, tracker_cls, include_answer=True)) for row in answerable]
    counterfactuals = [
        (row, run_window(row, tracker_cls, include_answer=False))
        for row in answerable
    ]
    no_silver_actuals = [
        (row, run_window(row, tracker_cls, include_answer=True))
        for row in no_silver
    ]

    exact = sum(
        1
        for row, actual in actuals
        if actual["status"] == "answered"
        and actual["answered_by_utterance_id"] == row["silver_answer"]["utterance"]["id"]
    )
    status = sum(1 for _, actual in actuals if actual["status"] == "answered")
    cf_unanswered = sum(1 for _, actual in counterfactuals if actual["status"] == "unanswered")
    no_silver_unanswered = sum(
        1 for _, actual in no_silver_actuals if actual["status"] == "unanswered"
    )

    return {
        "n": len(rows),
        "answerable": len(answerable),
        "no_silver": len(no_silver),
        "exact": exact,
        "status": status,
        "cf_unanswered": cf_unanswered,
        "no_silver_unanswered": no_silver_unanswered,
    }


def pct(num, denom):
    if denom == 0:
        return "NA"
    return f"{num / denom:.1%}"


def print_inventory(corpus_map):
    print("INVENTORY")
    for name, paths in corpus_map.items():
        rows = list(iter_question_rows(paths))
        speakers = Counter(row["speaker"] for row in rows)
        answerable = sum(1 for row in rows if row["silver_answer"] is not None)
        raw_short = sum(
            1
            for row in rows
            if row["silver_answer"] and row["silver_answer"]["kind"] == "raw_short_answer"
        )
        print(
            f"{name}\tfiles={len(paths)}\tquestions={len(rows)}"
            f"\tdoctor={speakers.get('doctor', 0)}"
            f"\tpatient={speakers.get('patient', 0)}"
            f"\tanswerable={answerable}\traw_short={raw_short}"
        )


def print_broad_probe(corpus_map):
    print("\nBROAD PROBE, PATIENT QUESTIONS ONLY")
    print(
        "corpus\ttracker\tthreshold\tn\tanswerable\tno_silver"
        "\texact\tstatus\tcf_unanswered\tno_silver_unanswered"
    )
    for corpus_name in ["challenge_aci", "challenge_eval", "src_experiment_aci", "all_json_annotations"]:
        rows = list(iter_question_rows(corpus_map[corpus_name], speaker_filter="patient"))
        trackers = [
            ("lexical", "-", Solution2Tracker),
            ("hybrid_raw", "0.45", tracker_with_threshold(CachedHybridTracker, 0.45)),
            ("hybrid_raw", "0.60", tracker_with_threshold(CachedHybridTracker, 0.60)),
            ("hybrid_no_raw", "0.45", tracker_with_threshold(CachedNoRawHybridTracker, 0.45)),
            ("hybrid_no_raw", "0.60", tracker_with_threshold(CachedNoRawHybridTracker, 0.60)),
        ]
        for tracker_name, threshold, tracker_cls in trackers:
            metrics = evaluate(rows, tracker_cls)
            print(
                f"{corpus_name}\t{tracker_name}\t{threshold}"
                f"\t{metrics['n']}\t{metrics['answerable']}\t{metrics['no_silver']}"
                f"\t{metrics['exact']}/{metrics['answerable']} {pct(metrics['exact'], metrics['answerable'])}"
                f"\t{metrics['status']}/{metrics['answerable']} {pct(metrics['status'], metrics['answerable'])}"
                f"\t{metrics['cf_unanswered']}/{metrics['answerable']} {pct(metrics['cf_unanswered'], metrics['answerable'])}"
                f"\t{metrics['no_silver_unanswered']}/{metrics['no_silver']} {pct(metrics['no_silver_unanswered'], metrics['no_silver'])}"
            )


def print_dual_simulation(corpus_map):
    rows = list(iter_question_rows(corpus_map["all_json_annotations"], speaker_filter="patient"))
    answerable = [row for row in rows if row["silver_answer"] is not None]
    no_silver = [row for row in rows if row["silver_answer"] is None]

    print("\nDUAL-THRESHOLD POST-HOC SIMULATION, PATIENT QUESTIONS, ALL JSON ANNOTATIONS")
    for label, base_cls in [
        ("hybrid_raw", CachedHybridTracker),
        ("hybrid_no_raw", CachedNoRawHybridTracker),
    ]:
        low_cls = tracker_with_threshold(base_cls, 0.45)
        high_cls = tracker_with_threshold(base_cls, 0.60)

        actual_buckets = Counter()
        actual_exact = Counter()
        for row in answerable:
            low = run_window(row, low_cls, include_answer=True)
            high = run_window(row, high_cls, include_answer=True)
            if high["status"] == "answered":
                bucket = "confirmed_answered"
                answer_id = high["answered_by_utterance_id"]
            elif low["status"] == "answered":
                bucket = "uncertain_candidate"
                answer_id = low["answered_by_utterance_id"]
            else:
                bucket = "still_open"
                answer_id = None
            actual_buckets[bucket] += 1
            if answer_id == row["silver_answer"]["utterance"]["id"]:
                actual_exact[bucket] += 1

        counterfactual_buckets = Counter()
        for row in answerable:
            low = run_window(row, low_cls, include_answer=False)
            high = run_window(row, high_cls, include_answer=False)
            if high["status"] == "answered":
                bucket = "false_confirmed_answered"
            elif low["status"] == "answered":
                bucket = "uncertain_candidate"
            else:
                bucket = "confirmed_unanswered"
            counterfactual_buckets[bucket] += 1

        no_silver_buckets = Counter()
        for row in no_silver:
            low = run_window(row, low_cls, include_answer=True)
            high = run_window(row, high_cls, include_answer=True)
            if high["status"] == "answered":
                bucket = "false_confirmed_answered"
            elif low["status"] == "answered":
                bucket = "uncertain_candidate"
            else:
                bucket = "confirmed_unanswered"
            no_silver_buckets[bucket] += 1

        confirmed = actual_buckets["confirmed_answered"]
        candidates = actual_buckets["uncertain_candidate"]
        cf_unanswered = counterfactual_buckets["confirmed_unanswered"]
        cf_not_confirmed = (
            counterfactual_buckets["confirmed_unanswered"]
            + counterfactual_buckets["uncertain_candidate"]
        )

        print(f"\n{label} low=0.45 high=0.60")
        print(f"answerable_total={len(answerable)} no_silver_total={len(no_silver)}")
        print(f"actual_buckets={dict(actual_buckets)}")
        print(f"actual_exact_within_answered_buckets={dict(actual_exact)}")
        print(f"counterfactual_buckets={dict(counterfactual_buckets)}")
        print(f"no_silver_buckets={dict(no_silver_buckets)}")
        print(f"confirmed_answered={confirmed}/{len(answerable)} {pct(confirmed, len(answerable))}")
        print(
            "candidate_or_confirmed_answered="
            f"{confirmed + candidates}/{len(answerable)} "
            f"{pct(confirmed + candidates, len(answerable))}"
        )
        print(
            "counterfactual_confirmed_unanswered="
            f"{cf_unanswered}/{len(answerable)} {pct(cf_unanswered, len(answerable))}"
        )
        print(
            "counterfactual_not_confirmed_answered="
            f"{cf_not_confirmed}/{len(answerable)} {pct(cf_not_confirmed, len(answerable))}"
        )


def main():
    corpus_map = corpora()
    print(f"ATLAS_ROOT={ATLAS_ROOT}")
    print_inventory(corpus_map)
    print_broad_probe(corpus_map)
    print_dual_simulation(corpus_map)


if __name__ == "__main__":
    main()

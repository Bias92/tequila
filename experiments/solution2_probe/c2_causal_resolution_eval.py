#!/usr/bin/env python3
"""Marker-free causal evaluation for C2 question resolution.

This script evaluates question resolution as a causal evidence relation:

    An answer evidence set E is useful if the original window resolves the
    question, but the same window with E removed no longer resolves it.

Unlike the older counterfactual CSV column, this script does not rely on a
`[REMOVED silver answer]` marker. It removes the gold answer utterance lines
from the original window and then runs the same resolver on the edited window.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Iterable

from c2_evidence_resolution_linker import (
    DEFAULT_INPUT,
    LinkPrediction,
    gold_answer_ids,
    gold_status,
    predict_link,
    read_rows,
    weighted_percent,
    write_csv,
)


DEFAULT_PREFIX = Path("/Users/markov/Downloads/c2_causal_resolution")


def delete_answer_lines(window: str, answer_ids: set[str]) -> tuple[str, list[str]]:
    """Remove whole gold answer utterance lines from a local window."""
    kept: list[str] = []
    deleted: list[str] = []

    for raw in window.splitlines():
        match = re.match(r"^(line_\d+)\s+\[[^\]]+\]\s*(.*)$", raw.strip())
        if match and match.group(1) in answer_ids:
            deleted.append(raw)
            continue
        kept.append(raw)

    return "\n".join(kept), deleted


def predict_on_window(row: dict[str, str], window: str, threshold: float) -> LinkPrediction:
    row_copy = dict(row)
    row_copy["_causal_window"] = window
    return predict_link(row_copy, "_causal_window", threshold)


def evaluate(rows: Iterable[dict[str, str]], threshold: float) -> tuple[dict[str, object], list[dict[str, object]]]:
    answered_total = 0
    suff_status = 0
    suff_exact = 0
    necessity = 0
    causal_exact_and_necessary = 0
    natural_unanswered_total = 0
    natural_unanswered = 0
    missing_deletion_total = 0

    details: list[dict[str, object]] = []

    for row in rows:
        status = gold_status(row)
        if status == "Exclude" or not status:
            continue

        weight = int(row["dup_count"])
        actual = predict_on_window(row, row["window_actual"], threshold)

        if status == "Answered":
            answer_ids = gold_answer_ids(row)
            deleted_window, deleted_lines = delete_answer_lines(row["window_actual"], answer_ids)
            deleted = predict_on_window(row, deleted_window, threshold)

            answered_total += weight
            if actual.answered:
                suff_status += weight
            exact = actual.answer_id in answer_ids
            necessary = not deleted.answered
            if exact:
                suff_exact += weight
            if necessary:
                necessity += weight
            if exact and necessary:
                causal_exact_and_necessary += weight
            if not deleted_lines:
                missing_deletion_total += weight

            details.append(
                {
                    "unique_id": row["unique_id"],
                    "dup_count": weight,
                    "conversation_id": row["conversation_id"],
                    "question_utterance_id": row["question_utterance_id"],
                    "gold_status": status,
                    "gold_answer_ids": ";".join(sorted(answer_ids)),
                    "deleted_line_count": len(deleted_lines),
                    "actual_answered": actual.answered,
                    "actual_answer_id": actual.answer_id,
                    "actual_score": f"{actual.score:.4f}",
                    "actual_reason": actual.reason,
                    "actual_evidence": actual.evidence,
                    "deleted_answered": deleted.answered,
                    "deleted_answer_id": deleted.answer_id,
                    "deleted_score": f"{deleted.score:.4f}",
                    "deleted_reason": deleted.reason,
                    "deleted_evidence": deleted.evidence,
                    "sufficiency_exact": exact,
                    "necessity": necessary,
                    "causal_success": exact and necessary,
                }
            )

        elif status == "Unanswered":
            natural_unanswered_total += weight
            unresolved = not actual.answered
            if unresolved:
                natural_unanswered += weight

            details.append(
                {
                    "unique_id": row["unique_id"],
                    "dup_count": weight,
                    "conversation_id": row["conversation_id"],
                    "question_utterance_id": row["question_utterance_id"],
                    "gold_status": status,
                    "gold_answer_ids": "",
                    "deleted_line_count": 0,
                    "actual_answered": actual.answered,
                    "actual_answer_id": actual.answer_id,
                    "actual_score": f"{actual.score:.4f}",
                    "actual_reason": actual.reason,
                    "actual_evidence": actual.evidence,
                    "deleted_answered": "",
                    "deleted_answer_id": "",
                    "deleted_score": "",
                    "deleted_reason": "",
                    "deleted_evidence": "",
                    "sufficiency_exact": "",
                    "necessity": "",
                    "causal_success": unresolved,
                }
            )

    summary: dict[str, object] = {
        "Method": f"CQRA causal deletion, thr={threshold:.2f}",
        "n_answered": answered_total,
        "n_unanswered": natural_unanswered_total,
        "Sufficiency status": weighted_percent(suff_status, answered_total),
        "Sufficiency status raw": f"{suff_status}/{answered_total}",
        "Sufficiency exact": weighted_percent(suff_exact, answered_total),
        "Sufficiency exact raw": f"{suff_exact}/{answered_total}",
        "Necessity after deletion": weighted_percent(necessity, answered_total),
        "Necessity after deletion raw": f"{necessity}/{answered_total}",
        "Causal exact+necessary": weighted_percent(causal_exact_and_necessary, answered_total),
        "Causal exact+necessary raw": f"{causal_exact_and_necessary}/{answered_total}",
        "Natural unanswered": weighted_percent(natural_unanswered, natural_unanswered_total),
        "Natural unanswered raw": f"{natural_unanswered}/{natural_unanswered_total}",
        "Missing deletion": missing_deletion_total,
    }
    return summary, details


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-prefix", type=Path, default=DEFAULT_PREFIX)
    parser.add_argument(
        "--thresholds",
        default="0.35,0.45,0.55,0.65,0.72,0.80,0.90,1.00",
        help="Comma-separated thresholds to sweep.",
    )
    parser.add_argument("--detail-threshold", type=float, default=0.45)
    args = parser.parse_args()

    rows = read_rows(args.input)
    thresholds = [float(value) for value in args.thresholds.split(",") if value.strip()]

    summaries: list[dict[str, object]] = []
    detail_rows: list[dict[str, object]] = []
    for threshold in thresholds:
        summary, details = evaluate(rows, threshold)
        summaries.append(summary)
        if abs(threshold - args.detail_threshold) < 1e-9:
            detail_rows = details

    summary_path = args.out_prefix.with_name(args.out_prefix.name + "_results.csv")
    detail_path = args.out_prefix.with_name(args.out_prefix.name + "_details.csv")
    write_csv(summary_path, summaries)
    if detail_rows:
        write_csv(detail_path, detail_rows)

    for summary in summaries:
        print(
            f"{summary['Method']}: "
            f"suff_exact={summary['Sufficiency exact']:.1f}% ({summary['Sufficiency exact raw']}), "
            f"necessity={summary['Necessity after deletion']:.1f}% ({summary['Necessity after deletion raw']}), "
            f"causal={summary['Causal exact+necessary']:.1f}% ({summary['Causal exact+necessary raw']}), "
            f"natural_unanswered={summary['Natural unanswered']:.1f}% ({summary['Natural unanswered raw']})"
        )
    print(f"Wrote {summary_path}")
    if detail_rows:
        print(f"Wrote {detail_path}")


if __name__ == "__main__":
    main()

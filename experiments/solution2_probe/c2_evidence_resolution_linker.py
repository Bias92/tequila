#!/usr/bin/env python3
"""Evaluate an evidence-constrained C2 question resolution linker.

This script is intentionally separate from the older LR edge scorer. It treats
question resolution as a local finite-state linking problem:

1. keep a patient question open,
2. scan later clinician utterances, stopping when the patient asks a new
   question (the open question's answer must precede the next question),
3. accept an answer edge only when the utterance contains explicit resolving
   evidence for the question type; once the patient has spoken again the
   adjacency pair is closed, so a later cue-only utterance must also share
   vocabulary with the question,
4. otherwise leave the question unresolved.

The goal is not to replace the manual gold sheet. It is a deterministic
experiment runner for the C2 manual-gold CSV.
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_INPUT = Path("/Users/markov/Downloads/c2_manual_gold_clean_aci_only_unique_window_latest.csv")
DEFAULT_PREFIX = Path("/Users/markov/Downloads/c2_evidence_resolution_linker")


STOPWORDS = set(
    """
    a an the and or but if then so to of in on for with without from by as at
    is are was were be been being am i you he she it we they me my your our
    their this that these those do does did done have has had having can could
    would should will shall may might must not no yes yeah okay ok right uh um
    kinda kind of like just really very also any about into onto than there here
    what when where why how who whom which nt don't i'm you're we're they're
    i've you've we'll i'll let's get got go going gonna want wanted need needed
    make made take taking tell told say said know think thought see saw look
    looked use used
    """.split()
)

LOW_INFORMATION = {
    "question",
    "questions",
    "concern",
    "concerns",
    "doctor",
    "patient",
    "thing",
    "things",
    "stuff",
    "something",
    "anything",
}

ACK_WORDS = {
    "okay",
    "ok",
    "right",
    "yeah",
    "yes",
    "no",
    "sure",
    "alright",
    "fine",
    "wonderful",
    "great",
    "absolutely",
    "correct",
    "mhm",
    "mm",
    "hmm",
}

ANSWER_CUE_PATTERNS = [
    r"\byes\b",
    r"\bno\b",
    r"\byeah\b",
    r"\bnot\b",
    r"\bdon'?t think\b",
    r"\bi think\b",
    r"\bi would\b",
    r"\byou should\b",
    r"\byou can\b",
    r"\byou cannot\b",
    r"\byou can't\b",
    r"\byou need\b",
    r"\bi recommend\b",
    r"\bavoid\b",
    r"\bcontinue\b",
    r"\bstop\b",
    r"\bstart\b",
    r"\btake\b",
    r"\bprescribe\b",
    r"\border\b",
    r"\brefer\b",
    r"\bschedule\b",
    r"\bfollow up\b",
    r"\bdiagnos",
    r"\bcondition\b",
    r"\bcause",
    r"\bbecause\b",
    r"\bdue to\b",
    r"\bconsistent with\b",
    r"\bfavorable\b",
    r"\bnormal\b",
    r"\babnormal\b",
    r"\bunder a year\b",
    r"\bweeks?\b",
    r"\bdays?\b",
    r"\bmonths?\b",
    r"\bmg\b",
    r"\bpercent\b",
    r"\b\d+\b",
]


@dataclass(frozen=True)
class WindowLine:
    utterance_id: str
    speaker: str
    text: str
    raw: str


@dataclass(frozen=True)
class LinkPrediction:
    answered: bool
    answer_id: str
    score: float
    evidence: str
    reason: str


def normalize(text: str) -> str:
    # "do n't"/"don't" -> "do not" so negation cues can match contractions
    text = text.lower().replace("n't", " not")
    text = re.sub(r"[^a-z0-9/%\.]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokens(text: str) -> list[str]:
    return [
        tok
        for tok in normalize(text).split()
        if len(tok) > 1 and tok not in STOPWORDS
    ]


def content_tokens(text: str) -> list[str]:
    return [tok for tok in tokens(text) if tok not in LOW_INFORMATION]


def parse_window(text: str) -> list[WindowLine]:
    lines: list[WindowLine] = []
    for raw in text.splitlines():
        raw = raw.strip()
        if not raw:
            continue
        match = re.match(r"^(line_\d+)\s+\[([^\]]+)\]\s*(.*)$", raw)
        if not match:
            continue
        utterance_id, speaker, body = match.groups()
        body = body.replace("[QUESTION]", "").replace("[SILVER_ANSWER]", "").strip()
        lines.append(WindowLine(utterance_id, speaker.lower(), body, raw))
    return lines


def question_types(question: str) -> set[str]:
    text = normalize(question)
    types: set[str] = set()

    if re.search(r"\b(can|could|should|need|do|does|did|is|are|am|was|were|will|would|may|might)\b", text):
        types.add("yesno")
    if text.endswith("right"):
        types.add("yesno")
    if re.search(r"\bwhen|how soon|how long|how often|how many|what time\b", text):
        types.add("time_qty")
    if re.search(r"\bwhat|which|where|who\b", text):
        types.add("what")
    if re.search(r"\bwhy|cause|caused|reason|trigger|triggered|due to|got to do\b", text):
        types.add("cause")
    if re.search(r"\b(medication|medicine|drug|take|prescribe|dose|ibuprofen|tylenol|pill|cla|chromium|honey|food|eat|drink)\b", text):
        types.add("treatment_safety")
    if re.search(r"\b(test|mri|xray|x ray|scan|lab|result|vital|blood pressure|glucose|sugar)\b", text):
        types.add("test_result")
    if re.search(r"\bdiagnos|syndrome|disease|condition|what was going on|what is it|have\b", text):
        types.add("diagnosis")
    if re.search(r"\b(play|ski|work|exercise|vacation|go|return|activity|basketball)\b", text):
        types.add("activity")
    if re.search(r"\b(call|appointment|schedule|scheduled|earlier|office|see you|follow up|follow-up|come in)\b", text):
        types.add("access_followup")
    if re.search(r"\bdie|prognosis|cancer|serious|worried|terrified\b", text):
        types.add("prognosis")

    return types or {"general"}


def is_acknowledgement(text: str) -> bool:
    words = normalize(text).split()
    if not words:
        return True
    if len(words) <= 4 and all(word in ACK_WORDS for word in words):
        return True
    return bool(re.fullmatch(r"(okay|ok|right|yeah|yes|no|sure|alright|great|correct|absolutely)[\s\.]*", normalize(text)))


def is_deferral(text: str) -> bool:
    return bool(
        re.search(
            r"\b(we('| wi)?ll get to that|come back to that|talk about that later|we can discuss that|hold on|not yet let's get|first let's|get to that)\b",
            normalize(text),
        )
    )


def is_clinician_question(text: str) -> bool:
    if "?" in text:
        return True
    return bool(
        re.search(
            r"\b(how long|how often|have you|did you|do you|are you|is it|was it|what about|anything else|tell me|when did|where is|does it|can you describe|what are|what were)\b",
            normalize(text),
        )
    )


def starts_as_clinician_question(text: str) -> bool:
    text_norm = normalize(text)
    return bool(
        re.search(
            r"^(okay so |ok so |alright |and |so |well )?(when you say|how long|how often|how frequently|have you|did you|do you|are you|is it|was it|what about|what are|what were|where is|when did|does it|can you describe|tell me)\b",
            text_norm,
        )
    )


def is_patient_question(text: str) -> bool:
    """Detect a new patient question. These transcripts carry no '?', so match
    interrogative word order instead of punctuation."""
    if "?" in text:
        return True
    text_norm = normalize(text)
    return bool(
        re.search(r"\b(can|could|should|do|does|will|would|am|is|are)\s+(i|we|it|that|this|there)\b", text_norm)
        or re.search(r"\bhow\s+(much|long|soon|often|about|many|do|can|should)\b", text_norm)
        or re.search(r"\bwhat\s+about\b|\bwhat'?s\s+a\b", text_norm)
        or re.search(r"\bwhen\s+(do|can|should|will)\b", text_norm)
        or re.search(r"\b(are|do|can|could|should|will|would)\s+you\b", text_norm)
        or re.search(r"\byou'?re\s+not\b", text_norm)
    )


def has_access_resolution(text: str) -> bool:
    return bool(
        re.search(
            r"\b(call|office|appointment|schedule|scheduled|follow up|follow-up|come in|see you|refer|referral|get you in|make an appointment|front office)\b",
            normalize(text),
        )
    )


def is_uncertain_nonanswer(text: str, qtypes: set[str]) -> bool:
    text_norm = normalize(text)
    uncertain = re.search(r"\b(i'?m unfamiliar|i do not know|i don't know|not sure|i'm not sure|unclear)\b", text_norm)
    resolving = re.search(r"\b(avoid|recommend|should|can|cannot|could|need|prescribe|refer|order|schedule|safe|okay|ok)\b", text_norm)
    return bool(uncertain and not resolving and "treatment_safety" in qtypes)


def cue_score(text: str, qtypes: set[str]) -> tuple[float, list[str]]:
    text_norm = normalize(text)
    score = 0.0
    reasons: list[str] = []

    cue_hits = sum(1 for pattern in ANSWER_CUE_PATTERNS if re.search(pattern, text_norm))
    if cue_hits:
        score += min(cue_hits, 5) * 0.18
        reasons.append(f"generic_cues={cue_hits}")

    def add_if(qtype: str, pattern: str, value: float, reason: str) -> None:
        nonlocal score
        if qtype in qtypes and re.search(pattern, text_norm):
            score += value
            reasons.append(reason)

    add_if("yesno", r"\b(yes|no|yeah|not|don'?t think|i think|should|can|can't|cannot|need|would)\b", 0.35, "yesno_resolution")
    add_if("time_qty", r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten|week|day|month|year|soon|daily|morning|evening|today|tomorrow)\b", 0.35, "time_or_quantity")
    add_if("treatment_safety", r"\b(avoid|should|can|can't|cannot|safe|okay|ok|recommend|prescribe|take|continue|stop|start|dose|under a year|not very good|harm)\b", 0.40, "treatment_or_safety")
    add_if("test_result", r"\b(result|blood pressure|\d+|normal|abnormal|mri|scan|x ray|xray|lab|schedule|preapproval|call|pressure|oxygen|fever)\b", 0.35, "test_or_result")
    add_if("diagnosis", r"\b(diagnosis|condition|syndrome|disease|think|don't think|consistent|caused|cause|hypersensitivity|pneumonitis|not)\b", 0.35, "diagnosis_resolution")
    add_if("activity", r"\b(can|can't|cannot|rest|avoid|play|ski|go|vacation|week|days|until|after|before|return)\b", 0.35, "activity_guidance")
    add_if("access_followup", r"\b(call|office|appointment|schedule|scheduled|follow up|follow-up|come in|see you|refer|referral|get you in|make an appointment|front office)\b", 0.45, "access_or_followup")
    add_if("prognosis", r"\b(prognosis|favorable|die|young|healthy|definitive|pathology|serious|worry|concern)\b", 0.35, "prognosis_resolution")

    return score, reasons


def has_direct_answer_evidence(text: str, qtypes: set[str], overlap: int) -> tuple[bool, str]:
    cue, cue_reasons = cue_score(text, qtypes)
    short_direct = len(normalize(text).split()) <= 8 and bool(re.search(r"\b(yes|no|yeah|not|sure|absolutely|correct)\b", normalize(text)))
    if "access_followup" in qtypes and not has_access_resolution(text):
        # heavy vocabulary overlap with the question can stand in for an
        # explicit access resolution
        if overlap < 4:
            return False, "missing_access_resolution"
    if cue >= 0.35:
        return True, "+".join(cue_reasons)
    if overlap >= 2:
        return True, f"overlap={overlap}"
    if short_direct:
        return True, "short_direct_answer"
    return False, "no_explicit_evidence"


def predict_link(row: dict[str, str], window_key: str, threshold: float) -> LinkPrediction:
    lines = parse_window(row[window_key])
    question_id = row["question_utterance_id"].strip()
    question = row["question_text"]
    qtypes = question_types(question)
    question_terms = set(content_tokens(question))

    try:
        question_index = next(i for i, line in enumerate(lines) if line.utterance_id == question_id)
    except StopIteration:
        question_index = 0

    patient_context: list[str] = []
    patient_intervened = False

    for index, line in enumerate(lines[question_index + 1 :], start=question_index + 1):
        if "patient" in line.speaker:
            if is_patient_question(line.text):
                break
            patient_intervened = True
            patient_context.extend(content_tokens(line.text))
            continue
        if "doctor" not in line.speaker and "clinician" not in line.speaker:
            continue
        if "[REMOVED silver answer]" in line.text:
            continue
        if is_acknowledgement(line.text) or is_deferral(line.text):
            continue
        if is_uncertain_nonanswer(line.text, qtypes):
            continue
        if starts_as_clinician_question(line.text):
            continue

        candidate_terms = set(content_tokens(line.text))
        context_terms = question_terms | set(patient_context)
        overlap = len(candidate_terms & context_terms)
        question_overlap = len(candidate_terms & question_terms)
        base = overlap / max(4, min(len(context_terms), 12))
        cue, cue_reasons = cue_score(line.text, qtypes)
        score = base + cue

        distance = index - question_index
        score -= max(0, distance - 3) * 0.03
        # adjacency-pair prior: the turn right after the question is where
        # direct answers live
        if distance == 1:
            score += 0.05
            cue_reasons.append("adjacency_bonus")

        short_direct = len(normalize(line.text).split()) <= 8 and bool(
            re.search(r"\b(yes|no|yeah|not|sure|absolutely|correct)\b", normalize(line.text))
        )
        if short_direct:
            score += 0.25
            cue_reasons.append("short_direct_answer")

        if is_clinician_question(line.text):
            score -= 0.25
            cue_reasons.append("question_penalty")

        # adjacency-pair completion: a substantive clinician turn right after
        # the question answers it whatever its surface form (state report,
        # definition, causal explanation). Guards: no counter-questions, at
        # least two content tokens, and scheduling questions still need an
        # access resolution.
        if (
            distance == 1
            and len(candidate_terms) >= 2
            and not is_clinician_question(line.text)
            and ("access_followup" not in qtypes or has_access_resolution(line.text))
        ):
            reason = ";".join(cue_reasons + ["adjacent_response", f"overlap={overlap}", f"distance={distance}"])
            return LinkPrediction(True, line.utterance_id, max(score, threshold), line.text, reason)

        has_evidence, evidence_reason = has_direct_answer_evidence(line.text, qtypes, overlap)
        if not has_evidence:
            continue
        # alignment gate: after the patient speaks again the adjacency pair is
        # closed — a cue-only line is the next topic, not this question's answer
        if patient_intervened and question_overlap == 0 and not short_direct:
            continue

        reason = ";".join(cue_reasons + [evidence_reason, f"overlap={overlap}", f"distance={distance}"])
        prediction = LinkPrediction(True, line.utterance_id, score, line.text, reason)
        if prediction.score >= threshold:
            return prediction

    return LinkPrediction(False, "", 0.0, "", "no_resolving_edge")


def gold_status(row: dict[str, str]) -> str:
    return row["GOLD_status"].strip()


def gold_answer_ids(row: dict[str, str]) -> set[str]:
    return {
        part.strip()
        for part in re.split(r"[;,]", row["GOLD_answer_utterance_id"])
        if part.strip()
    }


def weighted_percent(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator * 100


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def removed_marker_present(row: dict[str, str]) -> bool:
    return "[REMOVED silver answer]" in row["window_answer_removed"]


def predict_counterfactual(row: dict[str, str], threshold: float, removal_guard: bool) -> LinkPrediction:
    if removal_guard and removed_marker_present(row):
        return LinkPrediction(False, "", 0.0, "", "removal_guard")
    return predict_link(row, "window_answer_removed", threshold)


def visible_gold_answer_ids(row: dict[str, str], window_key: str) -> set[str]:
    visible_by_id: dict[str, bool] = {}
    for line in parse_window(row[window_key]):
        visible_by_id[line.utterance_id] = "[REMOVED silver answer]" not in line.text
    return {answer_id for answer_id in gold_answer_ids(row) if visible_by_id.get(answer_id, False)}


def evaluate(
    rows: Iterable[dict[str, str]],
    threshold: float,
    removal_guard: bool = False,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    answer_total = 0
    answer_status = 0
    exact_link = 0
    counterfactual_unanswered = 0
    no_answer_total = 0
    no_answer_unanswered = 0
    true_cf_total = 0
    true_cf_status = 0
    true_cf_exact_total = 0
    true_cf_exact = 0
    details: list[dict[str, object]] = []

    for row in rows:
        status = gold_status(row)
        if status == "Exclude" or not status:
            continue
        weight = int(row["dup_count"])

        actual = predict_link(row, "window_actual", threshold)
        counterfactual = predict_counterfactual(row, threshold, removal_guard)

        if status == "Answered":
            answer_total += weight
            if actual.answered:
                answer_status += weight
            if actual.answer_id in gold_answer_ids(row):
                exact_link += weight
            if not counterfactual.answered:
                counterfactual_unanswered += weight

            true_cf_total += weight
            remaining_answers = visible_gold_answer_ids(row, "window_answer_removed")
            if remaining_answers:
                true_cf_exact_total += weight
                if counterfactual.answered:
                    true_cf_status += weight
                if counterfactual.answer_id in remaining_answers:
                    true_cf_exact += weight
            else:
                if not counterfactual.answered:
                    true_cf_status += weight
        elif status == "Unanswered":
            no_answer_total += weight
            if not actual.answered:
                no_answer_unanswered += weight

            true_cf_total += weight
            if not actual.answered:
                true_cf_status += weight

        details.append(
            {
                "unique_id": row["unique_id"],
                "dup_count": weight,
                "conversation_id": row["conversation_id"],
                "question_utterance_id": row["question_utterance_id"],
                "gold_status": status,
                "gold_answer_ids": ";".join(sorted(gold_answer_ids(row))),
                "actual_answered": actual.answered,
                "actual_answer_id": actual.answer_id,
                "actual_score": f"{actual.score:.4f}",
                "actual_reason": actual.reason,
                "actual_evidence": actual.evidence,
                "counterfactual_answered": counterfactual.answered,
                "counterfactual_answer_id": counterfactual.answer_id,
                "counterfactual_score": f"{counterfactual.score:.4f}",
                "counterfactual_reason": counterfactual.reason,
                "counterfactual_evidence": counterfactual.evidence,
            }
        )

    method_name = "Evidence linker + removal guard" if removal_guard else "Evidence linker"
    summary: dict[str, object] = {
        "Method": f"{method_name}, thr={threshold:.2f}",
        "n": answer_total + no_answer_total,
        "Answer status": weighted_percent(answer_status, answer_total),
        "Answer status raw": f"{answer_status}/{answer_total}",
        "Exact link": weighted_percent(exact_link, answer_total),
        "Exact link raw": f"{exact_link}/{answer_total}",
        "Counterfactual unanswered": weighted_percent(counterfactual_unanswered, answer_total),
        "Counterfactual unanswered raw": f"{counterfactual_unanswered}/{answer_total}",
        "No-answer unanswered": weighted_percent(no_answer_unanswered, no_answer_total),
        "No-answer unanswered raw": f"{no_answer_unanswered}/{no_answer_total}",
        "Gold-aware CF status": weighted_percent(true_cf_status, true_cf_total),
        "Gold-aware CF status raw": f"{true_cf_status}/{true_cf_total}",
        "Gold-aware CF exact": weighted_percent(true_cf_exact, true_cf_exact_total),
        "Gold-aware CF exact raw": f"{true_cf_exact}/{true_cf_exact_total}",
    }
    return summary, details


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-prefix", type=Path, default=DEFAULT_PREFIX)
    parser.add_argument(
        "--thresholds",
        default="0.45,0.55,0.65,0.72,0.80,0.90,1.00,1.10,1.20",
        help="Comma-separated thresholds to sweep.",
    )
    parser.add_argument("--detail-threshold", type=float, default=0.80)
    parser.add_argument(
        "--no-removal-guard",
        action="store_true",
        help="Only emit the base evidence linker, without the counterfactual removal guard variant.",
    )
    args = parser.parse_args()

    rows = read_rows(args.input)
    thresholds = [float(value) for value in args.thresholds.split(",") if value.strip()]
    summaries = []
    detail_rows: list[dict[str, object]] = []

    for threshold in thresholds:
        for removal_guard in ([False] if args.no_removal_guard else [False, True]):
            summary, details = evaluate(rows, threshold, removal_guard=removal_guard)
            summaries.append(summary)
            if not removal_guard and abs(threshold - args.detail_threshold) < 1e-9:
                detail_rows = details

    summary_path = args.out_prefix.with_name(args.out_prefix.name + "_results.csv")
    detail_path = args.out_prefix.with_name(args.out_prefix.name + "_details.csv")
    write_csv(summary_path, summaries)
    if detail_rows:
        write_csv(detail_path, detail_rows)

    for summary in summaries:
        print(
            f"{summary['Method']}: "
            f"answer={summary['Answer status']:.1f}% ({summary['Answer status raw']}), "
            f"exact={summary['Exact link']:.1f}% ({summary['Exact link raw']}), "
            f"cf_unanswered={summary['Counterfactual unanswered']:.1f}% ({summary['Counterfactual unanswered raw']}), "
            f"no_answer={summary['No-answer unanswered']:.1f}% ({summary['No-answer unanswered raw']})"
        )
    print(f"Wrote {summary_path}")
    if detail_rows:
        print(f"Wrote {detail_path}")


if __name__ == "__main__":
    main()

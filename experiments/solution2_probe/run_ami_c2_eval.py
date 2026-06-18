from __future__ import annotations

import argparse
import glob
import json
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from run_matching_probe import CachedNoRawMatchingTracker


NITE = "{http://nite.sourceforge.net/}"
ELICIT_TYPES = {"el.inf", "el.sug", "el.ass", "el.und", "elicit"}
MINOR_TYPES = {"bck", "stl", "fra"}
TASK_TYPES = {"inf", "sug", "ass"}
MEANINGFUL_TYPES = {"inf", "sug", "ass", "off", "und", "be.pos", "be.neg"}


@dataclass
class Act:
    meeting: str
    act_id: str
    speaker: str
    start: float
    da_type: str
    text: str
    summary: str | None = None


def nite_id(element: ET.Element) -> str:
    return element.get(f"{NITE}id") or ""


def load_da_type_map(root: Path) -> dict[str, str]:
    tree = ET.parse(root / "ontologies" / "da-types.xml")
    mapping: dict[str, str] = {}
    for elem in tree.getroot().iter("da-type"):
        elem_id = nite_id(elem)
        name = elem.get("name")
        if elem_id and name:
            mapping[elem_id] = name
    return mapping


def load_words(root: Path, meeting: str, speaker: str) -> dict[str, tuple[str, float | None]]:
    path = root / "words" / f"{meeting}.{speaker}.words.xml"
    out: dict[str, tuple[str, float | None]] = {}
    if not path.exists():
        return out
    for word in ET.parse(path).getroot().iter("w"):
        start = word.get("starttime")
        out[nite_id(word)] = (word.text or "", float(start) if start else None)
    return out


def parse_word_range(href: str) -> tuple[int, int] | None:
    nums = re.findall(r"words(\d+)", href)
    if not nums:
        return None
    return int(nums[0]), int(nums[-1])


def load_meeting(root: Path, meeting: str, da_types: dict[str, str]) -> list[Act]:
    files = glob.glob(str(root / "dialogueActs" / f"{meeting}.*.dialog-act.xml"))
    speakers = sorted(
        {
            re.search(rf"{re.escape(meeting)}\.([A-Z])\.dialog-act", path).group(1)
            for path in files
        }
    )
    words = {speaker: load_words(root, meeting, speaker) for speaker in speakers}
    acts: list[Act] = []

    for speaker in speakers:
        path = root / "dialogueActs" / f"{meeting}.{speaker}.dialog-act.xml"
        if not path.exists():
            continue
        for dact in ET.parse(path).getroot().iter("dact"):
            da_type = "unlab"
            for pointer in dact.findall(f"{NITE}pointer"):
                if pointer.get("role") != "da-aspect":
                    continue
                match = re.search(r"id\(([^)]+)\)", pointer.get("href", ""))
                if match:
                    da_type = da_types.get(match.group(1), da_type)

            child = dact.find(f"{NITE}child")
            if child is None:
                continue
            word_range = parse_word_range(child.get("href", ""))
            if word_range is None:
                continue
            first, last = word_range
            tokens = [
                words[speaker].get(f"{meeting}.{speaker}.words{i}")
                for i in range(first, last + 1)
            ]
            tokens = [token for token in tokens if token]
            if not tokens:
                continue
            starts = [token[1] for token in tokens if token[1] is not None]
            text = " ".join(token[0] for token in tokens).strip()
            if not text or not starts:
                continue
            acts.append(
                Act(
                    meeting=meeting,
                    act_id=nite_id(dact),
                    speaker=speaker,
                    start=min(starts),
                    da_type=da_type,
                    text=text,
                )
            )
    acts.sort(key=lambda act: act.start)
    return acts


def load_adjacency(root: Path, meetings: Iterable[str]) -> tuple[dict[tuple[str, str], list[str]], set[tuple[str, str]], set[tuple[str, str]]]:
    source_to_targets: dict[tuple[str, str], list[str]] = {}
    sources: set[tuple[str, str]] = set()
    targets: set[tuple[str, str]] = set()
    for meeting in meetings:
        path = root / "dialogueActs" / f"{meeting}.adjacency-pairs.xml"
        if not path.exists():
            continue
        for pair in ET.parse(path).getroot().iter("adjacency-pair"):
            pair_sources: list[str] = []
            pair_targets: list[str] = []
            for pointer in pair.findall(f"{NITE}pointer"):
                match = re.search(r"id\(([^)]+)\)", pointer.get("href", ""))
                if not match:
                    continue
                act_id = match.group(1)
                if pointer.get("role") == "source":
                    pair_sources.append(act_id)
                    sources.add((meeting, act_id))
                elif pointer.get("role") == "target":
                    pair_targets.append(act_id)
                    targets.add((meeting, act_id))
            for source in pair_sources:
                source_to_targets.setdefault((meeting, source), []).extend(pair_targets)
    return source_to_targets, sources, targets


def load_summaries(path: Path) -> dict[tuple[str, str], str]:
    rows = json.loads(path.read_text())
    return {tuple(row["key"]): row["summary"] for row in rows}


def candidate_allowed(mode: str, act: Act) -> bool:
    if mode == "all":
        return True
    if mode == "no_minor":
        return act.da_type not in MINOR_TYPES
    if mode == "meaningful":
        return act.da_type in MEANINGFUL_TYPES
    if mode == "task":
        return act.da_type in TASK_TYPES
    raise ValueError(f"unknown candidate mode: {mode}")


def gold_label(
    mode: str,
    meeting: str,
    question: Act,
    question_index: int,
    index_by_id: dict[str, int],
    source_to_targets: dict[tuple[str, str], list[str]],
    sources: set[tuple[str, str]],
    targets: set[tuple[str, str]],
    window: int,
) -> bool | None:
    key = (meeting, question.act_id)
    if mode == "source":
        return key in sources
    if mode == "source_future":
        future_targets = [
            index_by_id[target_id]
            for target_id in source_to_targets.get(key, [])
            if target_id in index_by_id and index_by_id[target_id] > question_index
        ]
        if future_targets:
            return True
        if key in sources:
            return None
        return False
    if mode == "source_future_window":
        future_targets = [
            index_by_id[target_id]
            for target_id in source_to_targets.get(key, [])
            if target_id in index_by_id
            and question_index < index_by_id[target_id] <= question_index + window
        ]
        if future_targets:
            return True
        if key in sources:
            return None
        return False
    if mode == "strict_negative":
        future_targets = [
            index_by_id[target_id]
            for target_id in source_to_targets.get(key, [])
            if target_id in index_by_id and index_by_id[target_id] > question_index
        ]
        if future_targets:
            return True
        if key in sources or key in targets:
            return None
        return False
    raise ValueError(f"unknown gold mode: {mode}")


def run_question(
    acts: list[Act],
    question_index: int,
    summaries: dict[tuple[str, str], str],
    window: int,
    threshold: float,
    candidate_mode: str,
) -> str | None:
    question = acts[question_index]
    q_key = (question.meeting, question.act_id)
    q_summary = summaries.get(q_key)
    if q_summary is None:
        return None

    tracker = CachedNoRawMatchingTracker(
        min_edge_score=threshold,
        embedding_threshold=threshold,
        temporal_penalty=0.05,
    )
    tracker.update(
        {"id": question.act_id, "speaker": question.speaker, "text": q_summary},
        {"relevant": True, "detail_list": [{"type": "question", "summary": q_summary}]},
    )

    stop = min(len(acts), question_index + 1 + window)
    for candidate in acts[question_index + 1 : stop]:
        if candidate.speaker == question.speaker:
            continue
        if not candidate_allowed(candidate_mode, candidate):
            continue
        c_summary = summaries.get((candidate.meeting, candidate.act_id))
        if c_summary is None:
            continue
        tracker.update(
            {"id": candidate.act_id, "speaker": candidate.speaker, "text": c_summary},
            {"relevant": True, "detail_list": [{"type": "detail", "summary": c_summary}]},
        )

    tracker.finalize()
    for row in tracker.report():
        if row["utterance_id"] == question.act_id:
            return row["status"]
    return None


def score(
    acts_by_meeting: dict[str, list[Act]],
    summaries: dict[tuple[str, str], str],
    source_to_targets: dict[tuple[str, str], list[str]],
    sources: set[tuple[str, str]],
    targets: set[tuple[str, str]],
    window: int,
    threshold: float,
    candidate_mode: str,
    gold_mode: str,
) -> dict[str, float | int]:
    tp = fp = fn = tn = skipped = 0
    for meeting, acts in acts_by_meeting.items():
        index_by_id = {act.act_id: idx for idx, act in enumerate(acts)}
        for idx, act in enumerate(acts):
            if act.da_type not in ELICIT_TYPES:
                continue
            truth = gold_label(
                gold_mode,
                meeting,
                act,
                idx,
                index_by_id,
                source_to_targets,
                sources,
                targets,
                window,
            )
            if truth is None:
                skipped += 1
                continue
            predicted = run_question(acts, idx, summaries, window, threshold, candidate_mode)
            if predicted is None:
                skipped += 1
                continue
            pred_answered = predicted == "answered"
            if truth and pred_answered:
                tp += 1
            elif truth and not pred_answered:
                fn += 1
            elif not truth and pred_answered:
                fp += 1
            else:
                tn += 1
    n = tp + fp + fn + tn
    answered = tp + fn
    unanswered = tn + fp
    return {
        "n": n,
        "answered": answered,
        "unanswered": unanswered,
        "skipped": skipped,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "recall": tp / answered if answered else 0.0,
        "specificity": tn / unanswered if unanswered else 0.0,
        "accuracy": (tp + tn) / n if n else 0.0,
        "baseline_all_answered": answered / n if n else 0.0,
    }


def print_result(label: str, result: dict[str, float | int]) -> None:
    print(
        f"{label}\t"
        f"n={result['n']}\t"
        f"ans={result['answered']}\t"
        f"unans={result['unanswered']}\t"
        f"skip={result['skipped']}\t"
        f"TP={result['tp']}\tFP={result['fp']}\tFN={result['fn']}\tTN={result['tn']}\t"
        f"rec={result['recall']:.1%}\t"
        f"spec={result['specificity']:.1%}\t"
        f"acc={result['accuracy']:.1%}\t"
        f"all_ans={result['baseline_all_answered']:.1%}"
    )


def sample_label_audit(
    acts_by_meeting: dict[str, list[Act]],
    sources: set[tuple[str, str]],
    targets: set[tuple[str, str]],
    limit: int,
) -> None:
    print("\nLabel audit: not-source elicit samples")
    seen = 0
    for meeting, acts in acts_by_meeting.items():
        for idx, act in enumerate(acts):
            if act.da_type not in ELICIT_TYPES or (meeting, act.act_id) in sources:
                continue
            print(
                f"\n[{meeting}] {act.act_id} speaker={act.speaker} type={act.da_type} "
                f"target_only={(meeting, act.act_id) in targets}"
            )
            print(f"Q: {act.text}")
            for cand in acts[idx + 1 : min(len(acts), idx + 6)]:
                print(f"  + {cand.speaker} {cand.da_type}: {cand.text[:180]}")
            seen += 1
            if seen >= limit:
                return


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ami-root", default="/tmp/ami_check/ami")
    parser.add_argument("--summaries", default="/tmp/ami_summaries.json")
    parser.add_argument("--meetings", type=int, default=5)
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--audit", type=int, default=0)
    args = parser.parse_args()

    root = Path(args.ami_root)
    summaries = load_summaries(Path(args.summaries))
    da_types = load_da_type_map(root)
    meetings = sorted(
        re.search(r"/([^/]+)\.adjacency-pairs\.xml", path).group(1)
        for path in glob.glob(str(root / "dialogueActs" / "*.adjacency-pairs.xml"))
    )[: args.meetings]
    acts_by_meeting = {meeting: load_meeting(root, meeting, da_types) for meeting in meetings}
    source_to_targets, sources, targets = load_adjacency(root, meetings)

    total_elicit = sum(
        1 for acts in acts_by_meeting.values() for act in acts if act.da_type in ELICIT_TYPES
    )
    source_elicit = sum(
        1
        for meeting, acts in acts_by_meeting.items()
        for act in acts
        if act.da_type in ELICIT_TYPES and (meeting, act.act_id) in sources
    )
    target_only_elicit = sum(
        1
        for meeting, acts in acts_by_meeting.items()
        for act in acts
        if act.da_type in ELICIT_TYPES
        and (meeting, act.act_id) not in sources
        and (meeting, act.act_id) in targets
    )
    print(
        f"meetings={len(meetings)} acts={sum(len(x) for x in acts_by_meeting.values())} "
        f"elicit={total_elicit} source_elicit={source_elicit} "
        f"not_source={total_elicit - source_elicit} target_only_not_source={target_only_elicit} "
        f"summaries={len(summaries)} window={args.window}"
    )

    if args.audit:
        sample_label_audit(acts_by_meeting, sources, targets, args.audit)

    print("\nMain grid")
    for gold_mode in ["source", "source_future", "strict_negative"]:
        for candidate_mode in ["all", "no_minor", "meaningful", "task"]:
            for threshold in [0.30, 0.40, 0.50, 0.60]:
                result = score(
                    acts_by_meeting,
                    summaries,
                    source_to_targets,
                    sources,
                    targets,
                    args.window,
                    threshold,
                    candidate_mode,
                    gold_mode,
                )
                print_result(f"gold={gold_mode}\tcand={candidate_mode}\tthr={threshold:.2f}", result)


if __name__ == "__main__":
    main()

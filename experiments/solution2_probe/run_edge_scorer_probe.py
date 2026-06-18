from __future__ import annotations

import glob
import json
import math
import os
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from run_broad_dual_probe import (
    ANSWER_CANDIDATE_TYPES,
    CHALLENGE,
    LOOKAHEAD_UTTERANCES,
    SPLITS,
    clean_detail,
    corpora,
    find_silver_answer,
    is_raw_short_answer,
    iter_question_rows,
    load_docs,
    pct,
)
from run_ami_c2_eval import (
    ELICIT_TYPES,
    MEANINGFUL_TYPES,
    Act,
    candidate_allowed,
    load_adjacency,
    load_da_type_map,
    load_meeting,
    load_summaries,
)
from src.state_tracker_solution2 import Solution2Tracker


ATLAS_ROOT = Path(os.environ.get("ATLAS_ROOT", "/Users/markov/Desktop/atlas"))
CHALLENGE_ROOT = ATLAS_ROOT / "data" / "challenge_data"
AMI_ROOT = Path(os.environ.get("AMI_ROOT", "/tmp/ami_check/ami"))
AMI_SUMMARIES = Path(os.environ.get("AMI_SUMMARIES", "/tmp/ami_summaries.json"))


@dataclass
class Edge:
    q_key: tuple[str, str]
    q_summary: str
    q_speaker: str
    cand_key: tuple[str, str]
    cand_summary: str
    cand_text: str
    cand_type: str
    distance: int
    cand_rank: int
    label: int


def embedder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer("all-MiniLM-L6-v2")


class FeatureBuilder:
    def __init__(self):
        self._encoder = None
        self._emb_cache: dict[str, np.ndarray] = {}

    def _encode(self, text: str) -> np.ndarray:
        if self._encoder is None:
            self._encoder = embedder()
        if text not in self._emb_cache:
            self._emb_cache[text] = self._encoder.encode(text, normalize_embeddings=True)
        return self._emb_cache[text]

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return Solution2Tracker._tokens(text)

    def lexical(self, q: str, a: str) -> tuple[float, float, int]:
        qt = self._tokens(q)
        at = self._tokens(a)
        if not qt or not at:
            return 0.0, 0.0, 0
        overlap = qt & at
        containment = len(overlap) / min(len(qt), len(at))
        jaccard = len(overlap) / len(qt | at)
        return containment, jaccard, len(overlap)

    def cosine(self, q: str, a: str) -> float:
        qv = self._encode(q)
        av = self._encode(a)
        return float(np.dot(qv, av))

    def transform(self, edges: list[Edge]) -> np.ndarray:
        rows = []
        for e in edges:
            contain, jac, overlap = self.lexical(e.q_summary, e.cand_summary)
            emb = self.cosine(e.q_summary, e.cand_summary)
            words = len(re.findall(r"[A-Za-z0-9']+", e.cand_text.lower()))
            is_raw = 1.0 if e.cand_type == "raw_short" else 0.0
            is_detail = 1.0 if e.cand_type == "detail" else 0.0
            is_follow = 1.0 if e.cand_type == "follow_up" else 0.0
            is_med = 1.0 if e.cand_type == "medication" else 0.0
            is_social = 1.0 if e.cand_type == "social_history" else 0.0
            is_meaningful = 1.0 if e.cand_type in MEANINGFUL_TYPES else 0.0
            d = max(1, e.distance)
            r = max(1, e.cand_rank)
            rows.append(
                [
                    1.0 / d,
                    math.exp(-0.45 * (d - 1)),
                    min(d, 20) / 20.0,
                    1.0 / r,
                    contain,
                    jac,
                    min(overlap, 5) / 5.0,
                    emb,
                    is_raw,
                    1.0 if is_raw_short_answer(e.cand_text) else 0.0,
                    is_detail,
                    is_follow,
                    is_med,
                    is_social,
                    is_meaningful,
                    min(words, 20) / 20.0,
                ]
            )
        return np.asarray(rows, dtype=float)


def challenge_paths(split: str, kind: str = "aci") -> list[Path]:
    return sorted((CHALLENGE_ROOT / split / kind).glob("*.json"))


def iter_aci_rows(paths: Iterable[Path], speaker_filter: str = "patient"):
    for path, data in load_docs(paths):
        utterances = data["utterances"]
        ids = [utt["id"] for utt in utterances]
        utt_by_id = {utt["id"]: utt for utt in utterances}
        ann_by_id = {ann["utterance_id"]: ann for ann in data.get("annotations", [])}
        for ann in data.get("annotations", []):
            utt = utt_by_id.get(ann["utterance_id"])
            if not utt or utt.get("speaker") != speaker_filter:
                continue
            for detail in ann.get("detail_list") or []:
                if detail.get("type") != "question":
                    continue
                q_idx = ids.index(ann["utterance_id"])
                yield {
                    "path": path,
                    "split": path.parent.parent.name,
                    "utterances": utterances,
                    "ann_by_id": ann_by_id,
                    "question_index": q_idx,
                    "question_utterance": utt,
                    "question_detail": clean_detail(detail),
                    "silver_answer": find_silver_answer(q_idx, utterances, ids, utt_by_id, ann_by_id),
                }


def aci_candidate_edges(row, include_answer: bool = True) -> list[Edge]:
    utterances = row["utterances"]
    ann_by_id = row["ann_by_id"]
    q_idx = row["question_index"]
    q_utt = row["question_utterance"]
    q_detail = row["question_detail"]
    q_key = (str(row["path"]), q_utt["id"])
    skip_answer = None
    if not include_answer and row["silver_answer"] is not None:
        skip_answer = row["silver_answer"]["utterance"]["id"]
    edges: list[Edge] = []
    cand_rank = 0
    for idx in range(q_idx + 1, min(len(utterances), q_idx + LOOKAHEAD_UTTERANCES + 1)):
        utt = utterances[idx]
        if utt["speaker"] == q_utt["speaker"] or utt["id"] == skip_answer:
            continue
        ann = ann_by_id.get(utt["id"])
        made = False
        if ann:
            for detail in ann.get("detail_list") or []:
                if detail.get("type") not in ANSWER_CANDIDATE_TYPES:
                    continue
                cand_rank += 1
                label = int(
                    row["silver_answer"] is not None
                    and utt["id"] == row["silver_answer"]["utterance"]["id"]
                )
                edges.append(
                    Edge(
                        q_key=q_key,
                        q_summary=q_detail["summary"],
                        q_speaker=q_utt["speaker"],
                        cand_key=(str(row["path"]), utt["id"]),
                        cand_summary=detail["summary"],
                        cand_text=utt.get("text", ""),
                        cand_type=detail["type"],
                        distance=idx - q_idx,
                        cand_rank=cand_rank,
                        label=label,
                    )
                )
                made = True
        if is_raw_short_answer(utt.get("text", "")):
            cand_rank += 1
            label = int(
                row["silver_answer"] is not None
                and utt["id"] == row["silver_answer"]["utterance"]["id"]
            )
            edges.append(
                Edge(
                    q_key=q_key,
                    q_summary=q_detail["summary"],
                    q_speaker=q_utt["speaker"],
                    cand_key=(str(row["path"]), utt["id"]),
                    cand_summary=utt.get("text", ""),
                    cand_text=utt.get("text", ""),
                    cand_type="raw_short",
                    distance=idx - q_idx,
                    cand_rank=cand_rank,
                    label=label,
                )
            )
    return edges


def collect_aci_edges(rows, include_answer: bool = True) -> list[Edge]:
    out: list[Edge] = []
    for row in rows:
        out.extend(aci_candidate_edges(row, include_answer=include_answer))
    return out


def choose_threshold(model, fb: FeatureBuilder, rows, objective: str = "min_status_cf") -> tuple[float, dict]:
    thresholds = [i / 100 for i in range(5, 96, 2)]
    best_t = 0.5
    best_score = -1.0
    best_metrics = {}
    for t in thresholds:
        m = eval_aci_rows(rows, model, fb, t)
        exact = m["exact"] / m["answerable"] if m["answerable"] else 0.0
        status = m["status"] / m["answerable"] if m["answerable"] else 0.0
        cf = m["cf_unanswered"] / m["answerable"] if m["answerable"] else 0.0
        no = m["no_silver_unanswered"] / m["no_silver"] if m["no_silver"] else 0.0
        if objective == "min_status_cf":
            score = min(status, cf)
        elif objective == "balanced":
            score = 0.35 * exact + 0.35 * status + 0.2 * cf + 0.1 * no
        else:
            score = min(exact, cf)
        if score > best_score:
            best_score = score
            best_t = t
            best_metrics = m
    return best_t, best_metrics


def predict_row(edges: list[Edge], model, fb: FeatureBuilder, threshold: float | None):
    if not edges:
        return None, None
    scores = model.predict_proba(fb.transform(edges))[:, 1]
    best_i = int(np.argmax(scores))
    if threshold is not None and scores[best_i] < threshold:
        return None, float(scores[best_i])
    return edges[best_i].cand_key, float(scores[best_i])


def eval_aci_rows(rows, model, fb: FeatureBuilder, threshold: float) -> dict[str, int]:
    answerable = [r for r in rows if r["silver_answer"] is not None]
    no_silver = [r for r in rows if r["silver_answer"] is None]
    exact = status = cf_unanswered = no_silver_unanswered = 0
    for row in answerable:
        pred_key, _ = predict_row(aci_candidate_edges(row, True), model, fb, threshold)
        if pred_key is not None:
            status += 1
            if pred_key[1] == row["silver_answer"]["utterance"]["id"]:
                exact += 1
        pred_cf, _ = predict_row(aci_candidate_edges(row, False), model, fb, threshold)
        if pred_cf is None:
            cf_unanswered += 1
    for row in no_silver:
        pred_key, _ = predict_row(aci_candidate_edges(row, True), model, fb, threshold)
        if pred_key is None:
            no_silver_unanswered += 1
    return {
        "n": len(rows),
        "answerable": len(answerable),
        "no_silver": len(no_silver),
        "exact": exact,
        "status": status,
        "cf_unanswered": cf_unanswered,
        "no_silver_unanswered": no_silver_unanswered,
    }


def print_aci(label: str, metrics: dict, threshold: float):
    print(
        f"{label}\tthr={threshold:.2f}\tn={metrics['n']}\t"
        f"ans={metrics['answerable']}\tno={metrics['no_silver']}\t"
        f"exact={metrics['exact']}/{metrics['answerable']} {pct(metrics['exact'], metrics['answerable'])}\t"
        f"status={metrics['status']}/{metrics['answerable']} {pct(metrics['status'], metrics['answerable'])}\t"
        f"cf_unans={metrics['cf_unanswered']}/{metrics['answerable']} {pct(metrics['cf_unanswered'], metrics['answerable'])}\t"
        f"no_unans={metrics['no_silver_unanswered']}/{metrics['no_silver']} {pct(metrics['no_silver_unanswered'], metrics['no_silver'])}"
    )


def fit_models(train_edges: list[Edge], fb: FeatureBuilder):
    X = fb.transform(train_edges)
    y = np.asarray([e.label for e in train_edges])
    models = {
        "logreg": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced", C=0.8),
        ),
        "hgb": HistGradientBoostingClassifier(
            max_iter=180,
            learning_rate=0.04,
            l2_regularization=0.02,
            max_leaf_nodes=15,
            random_state=7,
        ),
    }
    for name, model in models.items():
        if name == "hgb":
            pos = max(1, int(y.sum()))
            neg = max(1, int((1 - y).sum()))
            weights = np.where(y == 1, neg / pos, 1.0)
            model.fit(X, y, sample_weight=weights)
        else:
            model.fit(X, y)
        try:
            auc = roc_auc_score(y, model.predict_proba(X)[:, 1])
        except Exception:
            auc = float("nan")
        print(f"trained\t{name}\tedges={len(y)}\tpos={int(y.sum())}\tauc_train={auc:.3f}")
    return models


def load_ami_rows(meeting_limit: int = 5, window: int = 20):
    da_types = load_da_type_map(AMI_ROOT)
    meetings = sorted(
        re.search(r"/([^/]+)\.adjacency-pairs\.xml", path).group(1)
        for path in glob.glob(str(AMI_ROOT / "dialogueActs" / "*.adjacency-pairs.xml"))
    )[:meeting_limit]
    summaries = load_summaries(AMI_SUMMARIES)
    acts_by = {m: load_meeting(AMI_ROOT, m, da_types) for m in meetings}
    source_to_targets, sources, _targets = load_adjacency(AMI_ROOT, meetings)
    rows = []
    for meeting, acts in acts_by.items():
        idx_by_id = {a.act_id: i for i, a in enumerate(acts)}
        for idx, act in enumerate(acts):
            if act.da_type not in ELICIT_TYPES:
                continue
            if (meeting, act.act_id) not in summaries:
                continue
            targets = [
                t
                for t in source_to_targets.get((meeting, act.act_id), [])
                if t in idx_by_id and idx < idx_by_id[t] <= idx + window
            ]
            if not targets:
                continue
            rows.append((meeting, acts, idx, targets, summaries))
    return rows


def ami_edges_for_row(row, window: int = 20) -> list[Edge]:
    meeting, acts, q_idx, targets, summaries = row
    q = acts[q_idx]
    out: list[Edge] = []
    rank = 0
    for c in acts[q_idx + 1 : min(len(acts), q_idx + 1 + window)]:
        if c.speaker == q.speaker:
            continue
        if not candidate_allowed("meaningful", c):
            continue
        if (meeting, c.act_id) not in summaries:
            continue
        rank += 1
        out.append(
            Edge(
                q_key=(meeting, q.act_id),
                q_summary=summaries[(meeting, q.act_id)],
                q_speaker=q.speaker,
                cand_key=(meeting, c.act_id),
                cand_summary=summaries[(meeting, c.act_id)],
                cand_text=c.text,
                cand_type=c.da_type,
                distance=acts.index(c) - q_idx,
                cand_rank=rank,
                label=int(c.act_id in targets),
            )
        )
    return out


def eval_ami_link(rows, model=None, fb: FeatureBuilder | None = None, threshold: float | None = None):
    ok = 0
    total = 0
    none = 0
    bydist: dict[str, list[int]] = {"d1": [0, 0], "d2_3": [0, 0], "d4p": [0, 0]}
    for row in rows:
        meeting, acts, q_idx, targets, summaries = row
        idx_by = {a.act_id: i for i, a in enumerate(acts)}
        d = min(idx_by[t] - q_idx for t in targets)
        bucket = "d1" if d == 1 else "d2_3" if d <= 3 else "d4p"
        edges = ami_edges_for_row(row)
        pred_key = None
        if model is None:
            if edges:
                pred_key = edges[0].cand_key
        else:
            pred_key, _ = predict_row(edges, model, fb, threshold)
        hit = bool(pred_key and pred_key[1] in targets)
        ok += int(hit)
        total += 1
        none += int(pred_key is None)
        bydist[bucket][0] += int(hit)
        bydist[bucket][1] += 1
    return ok, total, none, bydist


def print_ami(label: str, rows, model=None, fb=None, threshold=None):
    ok, total, none, bydist = eval_ami_link(rows, model, fb, threshold)
    parts = [f"{label}", f"link={ok}/{total} {pct(ok,total)}", f"none={none}"]
    for b in ["d1", "d2_3", "d4p"]:
        h, n = bydist[b]
        parts.append(f"{b}={h}/{n} {pct(h,n)}")
    print("\t".join(parts))


def main() -> None:
    corpus_map = corpora()
    train_rows = list(iter_aci_rows(challenge_paths("train") + challenge_paths("valid")))
    valid_rows = list(iter_aci_rows(challenge_paths("valid")))
    test_rows = list(
        iter_aci_rows(
            challenge_paths("clinicalnlp_taskB_test1")
            + challenge_paths("clinicalnlp_taskC_test2")
            + challenge_paths("clef_taskC_test3")
        )
    )
    all_rows = list(iter_aci_rows([p for s in SPLITS for p in challenge_paths(s)]))
    eval_test_rows = list(
        iter_question_rows(
            [
                p
                for split in [
                    "clinicalnlp_taskB_test1",
                    "clinicalnlp_taskC_test2",
                    "clef_taskC_test3",
                ]
                for p in sorted((CHALLENGE_ROOT / split / "eval").glob("*.json"))
            ],
            speaker_filter="patient",
        )
    )
    src_rows = list(iter_question_rows(corpus_map["src_experiment_aci"], speaker_filter="patient"))
    broad_unseen_rows = test_rows + eval_test_rows + src_rows
    print(f"aci_rows train+valid={len(train_rows)} valid={len(valid_rows)} test123={len(test_rows)} all={len(all_rows)}")
    print(
        "broad_rows "
        f"eval_test123={len(eval_test_rows)} "
        f"src_experiment={len(src_rows)} "
        f"broad_unseen={len(broad_unseen_rows)}"
    )

    fb = FeatureBuilder()
    train_edges = collect_aci_edges(train_rows, include_answer=True)
    models = fit_models(train_edges, fb)

    for name, model in models.items():
        t, vm = choose_threshold(model, fb, valid_rows, objective="min_status_cf")
        print_aci(f"VALID\t{name}", vm, t)
        print_aci(f"TEST123\t{name}", eval_aci_rows(test_rows, model, fb, t), t)
        print_aci(f"ALL_CHALLENGE\t{name}", eval_aci_rows(all_rows, model, fb, t), t)
        print_aci(f"EVAL_TEST123\t{name}", eval_aci_rows(eval_test_rows, model, fb, t), t)
        print_aci(f"SRC_EXPERIMENT\t{name}", eval_aci_rows(src_rows, model, fb, t), t)
        print_aci(f"BROAD_UNSEEN\t{name}", eval_aci_rows(broad_unseen_rows, model, fb, t), t)

    print("\nAMI answer-link transfer, first 5 meetings")
    ami_rows = load_ami_rows(meeting_limit=5, window=20)
    print(f"ami_answer_link_rows={len(ami_rows)}")
    print_ami("nearest_substantive", ami_rows)
    for name, model in models.items():
        # For AMI answer-linking all examples are answerable, so argmax is the fair comparison.
        print_ami(f"{name}_argmax", ami_rows, model, fb, threshold=None)
        t, _ = choose_threshold(model, fb, valid_rows, objective="min_status_cf")
        print_ami(f"{name}_aci_threshold_{t:.2f}", ami_rows, model, fb, threshold=t)


if __name__ == "__main__":
    main()

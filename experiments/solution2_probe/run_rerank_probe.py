from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

from sentence_transformers import CrossEncoder

from run_broad_dual_probe import (
    ANSWER_CANDIDATE_TYPES,
    LOOKAHEAD_UTTERANCES,
    clean_detail,
    corpora,
    is_raw_short_answer,
    iter_question_rows,
    pct,
)


@dataclass
class Candidate:
    utterance_id: str
    text: str
    kind: str
    offset: int


@dataclass
class Decision:
    status: str
    answered_by_utterance_id: Optional[str]
    score: float
    margin: float


class PairReranker:
    """Question-answer pair reranker probe.

    This is not another cosine-threshold matcher. It first retrieves possible
    answer utterances from the streaming lookahead window, then scores each
    (question, candidate answer) pair jointly with a cross-encoder. If the best
    candidate is not strong enough, or is not separated from the runner-up, the
    question stays open and finalizes as unanswered.
    """

    _model: Optional[CrossEncoder] = None
    _score_cache: dict[tuple[str, str], float] = {}

    def __init__(
        self,
        threshold: float,
        margin: float = 0.0,
        include_raw_short: bool = False,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        self.threshold = threshold
        self.margin = margin
        self.include_raw_short = include_raw_short
        self.model_name = model_name

    @classmethod
    def model(cls, model_name: str) -> CrossEncoder:
        if cls._model is None:
            cls._model = CrossEncoder(model_name, max_length=256)
        return cls._model

    def decide(self, question: str, candidates: List[Candidate]) -> Decision:
        if not candidates:
            return Decision("unanswered", None, float("-inf"), float("inf"))

        scores = self._scores(question, [c.text for c in candidates])
        ranked = sorted(zip(candidates, scores), key=lambda item: item[1], reverse=True)
        best_candidate, best_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else float("-inf")
        margin = best_score - second_score

        if best_score >= self.threshold and margin >= self.margin:
            return Decision("answered", best_candidate.utterance_id, best_score, margin)
        return Decision("unanswered", None, best_score, margin)

    def _scores(self, question: str, candidate_texts: List[str]) -> List[float]:
        missing = []
        missing_pairs = []
        for text in candidate_texts:
            key = (question, text)
            if key not in self._score_cache:
                missing.append(key)
                missing_pairs.append([question, text])
        if missing_pairs:
            model = self.model(self.model_name)
            values = model.predict(missing_pairs, show_progress_bar=False)
            for key, value in zip(missing, values):
                self._score_cache[key] = float(value)
        return [self._score_cache[(question, text)] for text in candidate_texts]


def candidate_texts(row, include_answer: bool, include_raw_short: bool) -> List[Candidate]:
    q_idx = row["question_index"]
    end = min(len(row["utterances"]), q_idx + LOOKAHEAD_UTTERANCES + 1)
    skip_answer_id = None
    if not include_answer and row["silver_answer"] is not None:
        skip_answer_id = row["silver_answer"]["utterance"]["id"]

    candidates: List[Candidate] = []
    question_speaker = row["question_utterance"]["speaker"]
    for idx in range(q_idx + 1, end):
        utt = row["utterances"][idx]
        if utt["id"] == skip_answer_id:
            continue
        if utt.get("speaker") == question_speaker:
            continue

        ann = row["ann_by_id"].get(utt["id"])
        if ann:
            for detail in ann.get("detail_list") or []:
                detail = clean_detail(detail)
                if detail["type"] in ANSWER_CANDIDATE_TYPES:
                    candidates.append(
                        Candidate(
                            utterance_id=utt["id"],
                            text=detail["summary"],
                            kind=detail["type"],
                            offset=idx - q_idx,
                        )
                    )

        if include_raw_short and is_raw_short_answer(utt.get("text", "")):
            candidates.append(
                Candidate(
                    utterance_id=utt["id"],
                    text=utt.get("text", ""),
                    kind="raw_short",
                    offset=idx - q_idx,
                )
            )
    return candidates


def evaluate(rows: Iterable[dict], reranker: PairReranker) -> dict:
    rows = list(rows)
    answerable = [row for row in rows if row["silver_answer"] is not None]
    no_silver = [row for row in rows if row["silver_answer"] is None]

    actuals = []
    for row in answerable:
        decision = reranker.decide(
            row["question_detail"]["summary"],
            candidate_texts(row, include_answer=True, include_raw_short=reranker.include_raw_short),
        )
        actuals.append((row, decision))

    counterfactuals = []
    for row in answerable:
        decision = reranker.decide(
            row["question_detail"]["summary"],
            candidate_texts(row, include_answer=False, include_raw_short=reranker.include_raw_short),
        )
        counterfactuals.append((row, decision))

    no_silver_actuals = []
    for row in no_silver:
        decision = reranker.decide(
            row["question_detail"]["summary"],
            candidate_texts(row, include_answer=True, include_raw_short=reranker.include_raw_short),
        )
        no_silver_actuals.append((row, decision))

    exact = sum(
        1
        for row, decision in actuals
        if decision.status == "answered"
        and decision.answered_by_utterance_id == row["silver_answer"]["utterance"]["id"]
    )
    status = sum(1 for _, decision in actuals if decision.status == "answered")
    cf_unanswered = sum(1 for _, decision in counterfactuals if decision.status == "unanswered")
    no_silver_unanswered = sum(1 for _, decision in no_silver_actuals if decision.status == "unanswered")
    return {
        "n": len(rows),
        "answerable": len(answerable),
        "no_silver": len(no_silver),
        "exact": exact,
        "status": status,
        "cf_unanswered": cf_unanswered,
        "no_silver_unanswered": no_silver_unanswered,
    }


def main():
    rows = list(iter_question_rows(corpora()["all_json_annotations"], speaker_filter="patient"))
    configs = []
    for include_raw in (False, True):
        for threshold in (-10.0, -8.0, -6.0, -4.0, -2.0, 0.0, 1.0):
            configs.append((include_raw, threshold, 0.0))
        for threshold in (-6.0, -4.0, -2.0, 0.0):
            configs.append((include_raw, threshold, 1.0))

    print("tracker\tsetting\tn\tanswerable\tno_silver\texact\tstatus\tcf_unanswered\tno_silver_unanswered")
    for include_raw, threshold, margin in configs:
        reranker = PairReranker(
            threshold=threshold,
            margin=margin,
            include_raw_short=include_raw,
        )
        metrics = evaluate(rows, reranker)
        setting = f"thr={threshold:g} margin={margin:g} raw={'on' if include_raw else 'off'}"
        print(
            f"rerank_abstain\t{setting}\t{metrics['n']}\t{metrics['answerable']}\t{metrics['no_silver']}"
            f"\t{metrics['exact']}/{metrics['answerable']} {pct(metrics['exact'], metrics['answerable'])}"
            f"\t{metrics['status']}/{metrics['answerable']} {pct(metrics['status'], metrics['answerable'])}"
            f"\t{metrics['cf_unanswered']}/{metrics['answerable']} {pct(metrics['cf_unanswered'], metrics['answerable'])}"
            f"\t{metrics['no_silver_unanswered']}/{metrics['no_silver']} {pct(metrics['no_silver_unanswered'], metrics['no_silver'])}"
        )


if __name__ == "__main__":
    main()

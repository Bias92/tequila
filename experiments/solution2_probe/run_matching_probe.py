from __future__ import annotations

from run_broad_dual_probe import corpora, evaluate, iter_question_rows, pct
from src.state_tracker_solution2_matching import (
    MatchingSolution2Tracker,
    NoRawMatchingSolution2Tracker,
)


class CachedMatchingTracker(MatchingSolution2Tracker):
    _global_embedding_cache = {}

    def _encode(self, text: str):
        cache = CachedMatchingTracker._global_embedding_cache
        if text not in cache:
            cache[text] = self._get_encoder().encode(text, normalize_embeddings=True)
        return cache[text]


class CachedNoRawMatchingTracker(NoRawMatchingSolution2Tracker):
    _global_embedding_cache = {}

    def _encode(self, text: str):
        cache = CachedNoRawMatchingTracker._global_embedding_cache
        if text not in cache:
            cache[text] = self._get_encoder().encode(text, normalize_embeddings=True)
        return cache[text]


def tracker_cls(base_cls, **kwargs):
    class Tracker(base_cls):
        def __init__(self):
            super().__init__(**kwargs)

    return Tracker


def main():
    rows = list(iter_question_rows(corpora()["all_json_annotations"], speaker_filter="patient"))
    configs = [
        (
            "matching_raw",
            "min=0.45 emb=0.45",
            tracker_cls(CachedMatchingTracker, min_edge_score=0.45, embedding_threshold=0.45),
        ),
        (
            "matching_raw",
            "min=0.50 emb=0.50",
            tracker_cls(CachedMatchingTracker, min_edge_score=0.50, embedding_threshold=0.50),
        ),
        (
            "matching_raw",
            "min=0.55 emb=0.55",
            tracker_cls(CachedMatchingTracker, min_edge_score=0.55, embedding_threshold=0.55),
        ),
        (
            "matching_raw",
            "min=0.60 emb=0.60",
            tracker_cls(CachedMatchingTracker, min_edge_score=0.60, embedding_threshold=0.60),
        ),
        (
            "matching_no_raw",
            "min=0.40 emb=0.40 pen=0.05",
            tracker_cls(
                CachedNoRawMatchingTracker,
                min_edge_score=0.40,
                embedding_threshold=0.40,
                temporal_penalty=0.05,
            ),
        ),
        (
            "matching_no_raw",
            "min=0.45 emb=0.45",
            tracker_cls(CachedNoRawMatchingTracker, min_edge_score=0.45, embedding_threshold=0.45),
        ),
        (
            "matching_no_raw",
            "min=0.50 emb=0.50",
            tracker_cls(CachedNoRawMatchingTracker, min_edge_score=0.50, embedding_threshold=0.50),
        ),
        (
            "matching_no_raw",
            "min=0.55 emb=0.55",
            tracker_cls(CachedNoRawMatchingTracker, min_edge_score=0.55, embedding_threshold=0.55),
        ),
        (
            "matching_no_raw",
            "min=0.60 emb=0.60",
            tracker_cls(CachedNoRawMatchingTracker, min_edge_score=0.60, embedding_threshold=0.60),
        ),
        (
            "matching_multi_raw",
            "min=0.55 emb=0.55 multi",
            tracker_cls(
                CachedMatchingTracker,
                min_edge_score=0.55,
                embedding_threshold=0.55,
                allow_multi_question_per_answer=True,
            ),
        ),
    ]

    print("tracker\tsetting\tn\tanswerable\tno_silver\texact\tstatus\tcf_unanswered\tno_silver_unanswered")
    for name, setting, cls in configs:
        metrics = evaluate(rows, cls)
        print(
            f"{name}\t{setting}\t{metrics['n']}\t{metrics['answerable']}\t{metrics['no_silver']}"
            f"\t{metrics['exact']}/{metrics['answerable']} {pct(metrics['exact'], metrics['answerable'])}"
            f"\t{metrics['status']}/{metrics['answerable']} {pct(metrics['status'], metrics['answerable'])}"
            f"\t{metrics['cf_unanswered']}/{metrics['answerable']} {pct(metrics['cf_unanswered'], metrics['answerable'])}"
            f"\t{metrics['no_silver_unanswered']}/{metrics['no_silver']} {pct(metrics['no_silver_unanswered'], metrics['no_silver'])}"
        )


if __name__ == "__main__":
    main()

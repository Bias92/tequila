from __future__ import annotations

from run_broad_dual_probe import corpora, evaluate, iter_question_rows, pct
from run_matching_probe import tracker_cls
from src.state_tracker_solution2 import Solution2Tracker
from src.state_tracker_solution2_hybrid import HybridSolution2Tracker
from src.state_tracker_solution2_matching import NoRawMatchingSolution2Tracker
from src.state_tracker_solution2_evidence import CachedEvidenceGatedTracker


class CachedHybridTracker(HybridSolution2Tracker):
    _global_embedding_cache = {}

    def _encode(self, text: str):
        cache = CachedHybridTracker._global_embedding_cache
        if text not in cache:
            cache[text] = self._get_encoder().encode(text, normalize_embeddings=True)
        return cache[text]


def main() -> None:
    rows = list(iter_question_rows(corpora()["all_json_annotations"], speaker_filter="patient"))
    configs = [
        ("lexical", "-", Solution2Tracker),
        ("hybrid_raw", "emb=0.60", tracker_cls(CachedHybridTracker, embedding_threshold=0.60)),
        (
            "matching_no_raw",
            "min=0.40 emb=0.40 pen=0.05",
            tracker_cls(
                NoRawMatchingSolution2Tracker,
                min_edge_score=0.40,
                embedding_threshold=0.40,
                temporal_penalty=0.05,
            ),
        ),
        (
            "evidence_gated",
            "min=0.44 emb=0.46 high=0.62 margin=0.06",
            CachedEvidenceGatedTracker,
        ),
        (
            "evidence_gated",
            "min=0.40 emb=0.44 high=0.60 margin=0.04",
            tracker_cls(
                CachedEvidenceGatedTracker,
                min_edge_score=0.40,
                embedding_threshold=0.44,
                high_embedding_threshold=0.60,
                margin=0.04,
            ),
        ),
        (
            "evidence_gated",
            "min=0.48 emb=0.48 high=0.64 margin=0.08",
            tracker_cls(
                CachedEvidenceGatedTracker,
                min_edge_score=0.48,
                embedding_threshold=0.48,
                high_embedding_threshold=0.64,
                margin=0.08,
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


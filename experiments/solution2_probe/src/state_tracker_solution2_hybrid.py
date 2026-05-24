from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from src.state_tracker_solution2 import ANSWER_CANDIDATE_TYPES, Solution2Tracker


YES_NO_MARKERS = {
    "yes", "yeah", "yep", "correct", "right", "exactly", "absolutely",
    "sure", "okay", "ok", "no", "nope", "not", "dont", "don't", "doesnt",
    "doesn't", "cant", "can't", "won't", "will", "can", "should",
}
ACK_ONLY = {"okay", "ok", "right", "mm", "mmhmm", "mhm", "uh", "huh"}


@dataclass
class HybridDecision:
    method: str
    score: float
    terms: List[str] = field(default_factory=list)


class HybridSolution2Tracker(Solution2Tracker):
    """C2 prototype that uses summaries plus raw utterance text.

    It still does not call an LLM. It combines:
    - short yes/no raw utterance handling,
    - lexical overlap,
    - optional sentence-transformer embedding similarity.

    This is closer to a plausible deterministic C2 than the tiny lexical-only
    prototype, but it is still a probe, not a production claim.
    """

    _shared_encoder = None

    def __init__(
        self,
        link_threshold: float = 0.16,
        answer_threshold: float = 0.24,
        embedding_threshold: float = 0.40,
        enable_embedding: bool = True,
    ):
        super().__init__(
            link_threshold=link_threshold,
            answer_threshold=answer_threshold,
        )
        self.embedding_threshold = embedding_threshold
        self.enable_embedding = enable_embedding
        self._embedding_cache: Dict[str, object] = {}

    def update(self, utterance: Dict, annotation: Optional[Dict]) -> None:
        # Even if C1 emits no annotation for a short raw answer like "yes it is",
        # C2 can still use the raw utterance to close a recent open question.
        if not annotation or not annotation.get("relevant"):
            self._try_raw_short_answer(utterance)
            return

        current_parent = None
        for detail in annotation.get("detail_list") or []:
            dtype = detail["type"]
            summary = detail["summary"]

            if dtype == "agenda_item":
                current_parent = self._create_item(utterance["id"], summary)
                continue

            parent = current_parent or self._find_parent(summary)
            if parent is None:
                parent = self._create_item(utterance["id"], summary)
            current_parent = parent
            parent.summaries.append(summary)

            if dtype == "question":
                self._open_question(parent, utterance, summary)
            elif dtype in ANSWER_CANDIDATE_TYPES:
                self._try_answer_questions(parent, utterance, summary)

        self._try_raw_short_answer(utterance)

    def _try_answer_questions(self, parent, utterance: Dict, summary: str) -> None:
        for question in self.questions:
            if question.status != "open":
                continue
            if question.speaker == utterance["speaker"]:
                continue

            same_parent = question.agenda_item_id == parent.id
            lexical_score, overlap = self._similarity(question.summary, summary)
            emb_score = self._embedding_similarity(question.summary, summary)

            decision: Optional[HybridDecision] = None
            if same_parent and lexical_score >= self.answer_threshold:
                decision = HybridDecision("same-parent lexical", lexical_score, sorted(overlap))
            elif emb_score >= self.embedding_threshold:
                decision = HybridDecision("embedding", emb_score, [])
            elif same_parent and emb_score >= self.embedding_threshold - 0.08:
                decision = HybridDecision("same-parent embedding", emb_score, [])

            if decision:
                self._mark_answered(question, utterance, summary, decision)

    def _try_raw_short_answer(self, utterance: Dict) -> None:
        text = utterance.get("text", "")
        if not self._is_short_answer(text):
            return

        for question in reversed(self.questions):
            if question.status != "open":
                continue
            if question.speaker == utterance["speaker"]:
                continue

            decision = HybridDecision("raw short answer", 1.0, [])
            self._mark_answered(question, utterance, text, decision)
            return

    def _mark_answered(self, question, utterance: Dict, answer_text: str, decision: HybridDecision) -> None:
        question.status = "answered"
        question.answered_by_utterance_id = utterance["id"]
        question.answered_by_summary = answer_text
        question.match_score = decision.score
        question.overlap_terms = [decision.method] + decision.terms

    def _find_parent(self, summary: str):
        best_item = None
        best_score = 0.0
        for item in self.agenda_items:
            lexical_score, _ = self._similarity(summary, item.topic)
            emb_score = self._embedding_similarity(summary, item.topic)
            score = max(lexical_score, emb_score)
            if score > best_score:
                best_score = score
                best_item = item
        if best_item and best_score >= self.link_threshold:
            return best_item
        return None

    @classmethod
    def _tokens(cls, text: str) -> Set[str]:
        tokens = super()._tokens(text)
        return {token for token in tokens if token not in ACK_ONLY}

    @staticmethod
    def _is_short_answer(text: str) -> bool:
        cleaned = []
        for ch in text.lower():
            cleaned.append(ch if ch.isalnum() or ch == "'" else " ")
        words = "".join(cleaned).split()
        if not words or len(words) > 8:
            return False
        token_set = set(words)
        if token_set <= ACK_ONLY:
            return False
        return bool(token_set & YES_NO_MARKERS)

    def _embedding_similarity(self, left: str, right: str) -> float:
        if not self.enable_embedding:
            return 0.0
        encoder = self._get_encoder()
        if encoder is None:
            return 0.0
        left_vec = self._encode(left)
        right_vec = self._encode(right)
        try:
            return float((left_vec * right_vec).sum())
        except Exception:
            return 0.0

    def _encode(self, text: str):
        if text not in self._embedding_cache:
            self._embedding_cache[text] = self._get_encoder().encode(
                text,
                normalize_embeddings=True,
            )
        return self._embedding_cache[text]

    def _get_encoder(self):
        if HybridSolution2Tracker._shared_encoder is not None:
            return HybridSolution2Tracker._shared_encoder
        try:
            from sentence_transformers import SentenceTransformer

            HybridSolution2Tracker._shared_encoder = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception:
            HybridSolution2Tracker._shared_encoder = None
        return HybridSolution2Tracker._shared_encoder

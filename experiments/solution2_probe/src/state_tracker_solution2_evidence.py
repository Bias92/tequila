from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Set

from src.state_tracker_solution2 import STOPWORDS
from src.state_tracker_solution2_matching import CandidateEdge, MatchingSolution2Tracker


QUESTION_WORDS = {
    "can", "could", "should", "would", "will", "do", "does", "did", "is",
    "are", "was", "were", "what", "when", "where", "why", "how", "which",
    "who", "whom", "whether", "if", "any", "anything", "question",
    "questions", "ask", "asks", "asking",
}

VAGUE_RESPONSE_TERMS = {
    "acknowledge", "acknowledges", "acknowledged", "agree", "agrees",
    "agreement", "okay", "ok", "right", "mm", "mhm", "uh", "huh",
    "continue", "continues", "continued", "move", "moves", "next",
    "discussion", "briefly", "laughs", "unclear",
}

ANSWER_CUE_TERMS = {
    "yes", "yeah", "yep", "no", "nope", "not", "safe", "okay", "ok",
    "can", "cannot", "can't", "should", "will", "recommend", "recommends",
    "increase", "decrease", "take", "stop", "start", "refer", "order",
    "plan", "because", "due", "cause", "means", "result", "normal",
    "abnormal", "negative", "positive",
}


@dataclass
class EvidenceConfig:
    min_edge_score: float = 0.44
    embedding_threshold: float = 0.46
    high_embedding_threshold: float = 0.62
    margin: float = 0.06
    temporal_penalty: float = 0.04
    same_parent_bonus: float = 0.06
    focus_bonus: float = 0.06
    cue_bonus: float = 0.04


class EvidenceGatedSolution2Tracker(MatchingSolution2Tracker):
    """Evidence-gated question tracker.

    This is intended as the next algorithmic step after the matching probe.

    The previous matching tracker collected question-answer edges and selected
    a high-scoring assignment. This variant adds three gates before a question
    can close:

    1. The later utterance must look like an answer candidate, not merely any
       backchannel / stall / vague continuation.
    2. It must carry either focus overlap with the question or strong semantic
       evidence.
    3. At finalize time, the best edge must be separated from the runner-up by
       a margin, otherwise the question remains open.

    It still uses the same low-cost lexical / embedding ingredients; the new
    part is the state update rule around those signals.
    """

    def __init__(self, **kwargs):
        cfg = EvidenceConfig()
        for key, value in kwargs.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)
        self.evidence_config = cfg
        super().__init__(
            embedding_threshold=cfg.embedding_threshold,
            min_edge_score=cfg.min_edge_score,
            temporal_penalty=cfg.temporal_penalty,
            same_parent_bonus=cfg.same_parent_bonus,
            allow_multi_question_per_answer=False,
        )

    def _focus_tokens(self, text: str) -> Set[str]:
        return {
            token
            for token in self._tokens(text)
            if token not in QUESTION_WORDS
            and token not in STOPWORDS
            and token not in VAGUE_RESPONSE_TERMS
        }

    def _answer_tokens(self, text: str) -> Set[str]:
        return {
            token
            for token in self._tokens(text)
            if token not in VAGUE_RESPONSE_TERMS
        }

    def _looks_like_answer_candidate(self, summary: str, raw_text: str = "") -> bool:
        tokens = self._answer_tokens(f"{summary} {raw_text}")
        if not tokens:
            return False

        cue_hit = bool(tokens & ANSWER_CUE_TERMS)
        if cue_hit:
            return True

        # Reject pure acknowledgements / discourse movement.
        vague_hits = self._tokens(f"{summary} {raw_text}") & VAGUE_RESPONSE_TERMS
        if len(tokens) <= 2 and vague_hits:
            return False

        # Substantive summaries usually have at least two non-vague content
        # tokens. This keeps AMI backchannels like "Speaker acknowledges" out.
        return len(tokens) >= 2

    def _collect_answer_edges(self, parent, utterance: Dict, summary: str, dtype: str) -> None:
        raw_text = utterance.get("text", "")
        if not self._looks_like_answer_candidate(summary, raw_text):
            return

        answer_focus = self._focus_tokens(summary)
        cfg = self.evidence_config

        for q_index, question in enumerate(self.questions):
            if question.status != "open":
                continue
            if question.speaker == utterance["speaker"]:
                continue

            q_focus = self._focus_tokens(question.summary)
            focus_overlap = q_focus & answer_focus
            same_parent = question.agenda_item_id == parent.id

            lexical_score, overlap = self._similarity(question.summary, summary)
            emb_score = self._embedding_similarity(question.summary, summary)

            # Main cross-domain guard: do not let generic semantic similarity
            # close a question unless it is either very strong or supported by
            # focus overlap / same agenda.
            has_grounding = bool(focus_overlap) or same_parent or emb_score >= cfg.high_embedding_threshold
            if not has_grounding:
                continue

            distance = max(1, self._utterance_order - self._question_order.get(question.utterance_id, 0))
            cue_bonus = cfg.cue_bonus if (answer_focus & ANSWER_CUE_TERMS) else 0.0
            focus_bonus = cfg.focus_bonus if focus_overlap else 0.0
            parent_bonus = cfg.same_parent_bonus if same_parent else 0.0

            score = (
                0.55 * emb_score
                + 0.45 * lexical_score
                + focus_bonus
                + parent_bonus
                + cue_bonus
                - cfg.temporal_penalty * (distance - 1)
            )

            if score < cfg.min_edge_score:
                continue

            method = "evidence"
            terms = sorted((overlap | focus_overlap))
            self._candidate_edges.append(
                CandidateEdge(
                    question_index=q_index,
                    answer_key=utterance["id"],
                    utterance_id=utterance["id"],
                    utterance_order=self._utterance_order,
                    speaker=utterance["speaker"],
                    answer_text=summary,
                    decision=self._make_decision(method, score, terms),
                )
            )

    def _collect_raw_short_edges(self, utterance: Dict) -> None:
        # Raw short answers are useful in ACI, but dangerous cross-domain.
        # Keep them only when they are immediate and the question is yes/no-like.
        text = utterance.get("text", "")
        if not self._is_short_answer(text):
            return
        cfg = self.evidence_config
        for q_index, question in enumerate(self.questions):
            if question.status != "open":
                continue
            if question.speaker == utterance["speaker"]:
                continue
            distance = max(1, self._utterance_order - self._question_order.get(question.utterance_id, 0))
            if distance > 2:
                continue
            q_tokens = self._tokens(question.summary)
            if not (q_tokens & QUESTION_WORDS):
                continue
            score = 0.72 - cfg.temporal_penalty * (distance - 1)
            if score < cfg.min_edge_score:
                continue
            self._candidate_edges.append(
                CandidateEdge(
                    question_index=q_index,
                    answer_key=utterance["id"],
                    utterance_id=utterance["id"],
                    utterance_order=self._utterance_order,
                    speaker=utterance["speaker"],
                    answer_text=text,
                    decision=self._make_decision("evidence raw short", score, []),
                )
            )

    def _make_decision(self, method: str, score: float, terms: List[str]):
        from src.state_tracker_solution2_hybrid import HybridDecision

        return HybridDecision(method, score, terms)

    def _apply_matching(self) -> None:
        edges_by_question: Dict[int, List[CandidateEdge]] = defaultdict(list)
        for edge in self._candidate_edges:
            edges_by_question[edge.question_index].append(edge)

        candidates: List[CandidateEdge] = []
        for q_idx, edges in edges_by_question.items():
            edges.sort(key=lambda edge: edge.decision.score, reverse=True)
            best = edges[0]
            second_score = edges[1].decision.score if len(edges) > 1 else 0.0
            if best.decision.score - second_score < self.evidence_config.margin:
                continue
            candidates.append(best)

        # One answer utterance should normally close at most one question. Pick
        # high-confidence closures first.
        used_answers: set[str] = set()
        used_questions: set[int] = set()
        for edge in sorted(candidates, key=lambda edge: edge.decision.score, reverse=True):
            if edge.answer_key in used_answers or edge.question_index in used_questions:
                continue
            used_answers.add(edge.answer_key)
            used_questions.add(edge.question_index)
            self._mark_edge_answered(edge)


class CachedEvidenceGatedTracker(EvidenceGatedSolution2Tracker):
    _global_embedding_cache = {}

    def _encode(self, text: str):
        cache = CachedEvidenceGatedTracker._global_embedding_cache
        if text not in cache:
            cache[text] = self._get_encoder().encode(text, normalize_embeddings=True)
        return cache[text]


from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from src.state_tracker_solution2 import ANSWER_CANDIDATE_TYPES, Solution2Tracker
from src.state_tracker_solution2_hybrid import HybridDecision, HybridSolution2Tracker


@dataclass
class CandidateEdge:
    question_index: int
    answer_key: str
    utterance_id: str
    utterance_order: int
    speaker: str
    answer_text: str
    decision: HybridDecision


class MatchingSolution2Tracker(HybridSolution2Tracker):
    """Delayed question-answer matching probe.

    The earlier hybrid tracker greedily closes a question as soon as one answer
    candidate crosses a threshold. This variant stores all candidate edges in
    the streaming window, then picks a maximum-scoring one-to-one assignment at
    finalize time.

    It is still a small probe, not production code. The point is to test
    whether a more global C2 decision rule improves the answer/unanswered
    trade-off on the same silver benchmark.
    """

    def __init__(
        self,
        link_threshold: float = 0.16,
        answer_threshold: float = 0.24,
        embedding_threshold: float = 0.50,
        enable_embedding: bool = True,
        min_edge_score: float = 0.50,
        raw_short_score: float = 0.78,
        same_parent_bonus: float = 0.08,
        type_bonus: float = 0.03,
        temporal_penalty: float = 0.03,
        allow_multi_question_per_answer: bool = False,
    ):
        super().__init__(
            link_threshold=link_threshold,
            answer_threshold=answer_threshold,
            embedding_threshold=embedding_threshold,
            enable_embedding=enable_embedding,
        )
        self.min_edge_score = min_edge_score
        self.raw_short_score = raw_short_score
        self.same_parent_bonus = same_parent_bonus
        self.type_bonus = type_bonus
        self.temporal_penalty = temporal_penalty
        self.allow_multi_question_per_answer = allow_multi_question_per_answer
        self._utterance_order = 0
        self._question_order: Dict[str, int] = {}
        self._candidate_edges: List[CandidateEdge] = []

    def update(self, utterance: Dict, annotation: Optional[Dict]) -> None:
        self._utterance_order += 1

        if not annotation or not annotation.get("relevant"):
            self._collect_raw_short_edges(utterance)
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
                before = len(self.questions)
                self._open_question(parent, utterance, summary)
                if len(self.questions) > before:
                    self._question_order[self.questions[-1].utterance_id] = self._utterance_order
            elif dtype in ANSWER_CANDIDATE_TYPES:
                self._collect_answer_edges(parent, utterance, summary, dtype)

        self._collect_raw_short_edges(utterance)

    def finalize(self) -> None:
        self._apply_matching()
        super().finalize()

    def _collect_answer_edges(self, parent, utterance: Dict, summary: str, dtype: str) -> None:
        for q_index, question in enumerate(self.questions):
            if question.status != "open":
                continue
            if question.speaker == utterance["speaker"]:
                continue

            same_parent = question.agenda_item_id == parent.id
            lexical_score, overlap = self._similarity(question.summary, summary)
            emb_score = self._embedding_similarity(question.summary, summary)
            distance = max(1, self._utterance_order - self._question_order.get(question.utterance_id, 0))
            score = max(lexical_score, emb_score)
            if same_parent:
                score += self.same_parent_bonus
            if dtype in {"follow_up", "medication"}:
                score += self.type_bonus
            score -= self.temporal_penalty * (distance - 1)

            if score < self.min_edge_score:
                continue

            if emb_score >= lexical_score:
                method = "matching embedding"
                method_score = emb_score
                terms = []
            else:
                method = "matching lexical"
                method_score = lexical_score
                terms = sorted(overlap)
            decision = HybridDecision(method, max(method_score, score), terms)
            self._candidate_edges.append(
                CandidateEdge(
                    question_index=q_index,
                    answer_key=utterance["id"],
                    utterance_id=utterance["id"],
                    utterance_order=self._utterance_order,
                    speaker=utterance["speaker"],
                    answer_text=summary,
                    decision=decision,
                )
            )

    def _collect_raw_short_edges(self, utterance: Dict) -> None:
        text = utterance.get("text", "")
        if not self._is_short_answer(text):
            return
        for q_index, question in enumerate(self.questions):
            if question.status != "open":
                continue
            if question.speaker == utterance["speaker"]:
                continue
            distance = max(1, self._utterance_order - self._question_order.get(question.utterance_id, 0))
            score = self.raw_short_score - self.temporal_penalty * (distance - 1)
            if score < self.min_edge_score:
                continue
            self._candidate_edges.append(
                CandidateEdge(
                    question_index=q_index,
                    answer_key=utterance["id"],
                    utterance_id=utterance["id"],
                    utterance_order=self._utterance_order,
                    speaker=utterance["speaker"],
                    answer_text=text,
                    decision=HybridDecision("matching raw short answer", score, []),
                )
            )

    def _apply_matching(self) -> None:
        if self.allow_multi_question_per_answer:
            self._apply_multi_close()
            return

        edges_by_question: Dict[int, List[CandidateEdge]] = {}
        for edge in self._candidate_edges:
            edges_by_question.setdefault(edge.question_index, []).append(edge)
        for edges in edges_by_question.values():
            edges.sort(key=lambda e: (e.decision.score, -e.utterance_order), reverse=True)

        q_indices = sorted(edges_by_question)
        best_score = float("-inf")
        best_edges: List[CandidateEdge] = []

        def search(pos: int, used_answers: set[str], chosen: List[CandidateEdge], score: float) -> None:
            nonlocal best_score, best_edges
            if pos == len(q_indices):
                if score > best_score:
                    best_score = score
                    best_edges = list(chosen)
                return

            q_idx = q_indices[pos]
            search(pos + 1, used_answers, chosen, score)
            for edge in edges_by_question[q_idx]:
                if edge.answer_key in used_answers:
                    continue
                used_answers.add(edge.answer_key)
                chosen.append(edge)
                search(pos + 1, used_answers, chosen, score + edge.decision.score)
                chosen.pop()
                used_answers.remove(edge.answer_key)

        search(0, set(), [], 0.0)
        for edge in best_edges:
            self._mark_edge_answered(edge)

    def _apply_multi_close(self) -> None:
        best_by_question: Dict[int, CandidateEdge] = {}
        for edge in self._candidate_edges:
            old = best_by_question.get(edge.question_index)
            if old is None or edge.decision.score > old.decision.score:
                best_by_question[edge.question_index] = edge
        for edge in best_by_question.values():
            self._mark_edge_answered(edge)

    def _mark_edge_answered(self, edge: CandidateEdge) -> None:
        question = self.questions[edge.question_index]
        if question.status != "open":
            return
        question.status = "answered"
        question.answered_by_utterance_id = edge.utterance_id
        question.answered_by_summary = edge.answer_text
        question.match_score = edge.decision.score
        question.overlap_terms = [edge.decision.method] + edge.decision.terms


class NoRawMatchingSolution2Tracker(MatchingSolution2Tracker):
    def _collect_raw_short_edges(self, utterance: Dict) -> None:
        return None

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set


ANSWER_CANDIDATE_TYPES = {"detail", "follow_up", "medication", "social_history"}
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "between", "by", "can", "for",
    "from", "has", "have", "her", "his", "if", "in", "is", "it", "of", "or",
    "patient", "she", "he", "the", "their", "to", "will", "with", "doctor",
    "asks", "ask", "about", "whether", "what", "when", "why", "how", "does",
    "do", "did", "reports", "confirms", "states", "acknowledges",
}


@dataclass
class QuestionState:
    question_id: str
    utterance_id: str
    speaker: str
    summary: str
    agenda_item_id: str
    status: str = "open"
    answered_by_utterance_id: Optional[str] = None
    answered_by_summary: Optional[str] = None
    match_score: float = 0.0
    overlap_terms: List[str] = field(default_factory=list)


@dataclass
class AgendaItem:
    id: str
    topic: str
    first_utterance_id: str
    summaries: List[str] = field(default_factory=list)
    questions: List[str] = field(default_factory=list)
    question_ids: List[str] = field(default_factory=list)


class Solution2Tracker:
    """Tiny C2 prototype for unanswered-question tracking.

    This intentionally does not use an LLM. It only uses C1-style summaries.
    The point is to test whether a simple deterministic tracker can handle
    clear cases and expose where the idea breaks.
    """

    def __init__(self, link_threshold: float = 0.18, answer_threshold: float = 0.20):
        self.link_threshold = link_threshold
        self.answer_threshold = answer_threshold
        self.agenda_items: List[AgendaItem] = []
        self.questions: List[QuestionState] = []

    def update(self, utterance: Dict, annotation: Optional[Dict]) -> None:
        if not annotation or not annotation.get("relevant"):
            return

        current_parent: Optional[AgendaItem] = None
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

    def finalize(self) -> None:
        for question in self.questions:
            if question.status == "open":
                question.status = "unanswered"

    def report(self) -> List[Dict]:
        return [
            {
                "utterance_id": q.utterance_id,
                "speaker": q.speaker,
                "summary": q.summary,
                "status": q.status,
                "answered_by_utterance_id": q.answered_by_utterance_id,
                "answered_by_summary": q.answered_by_summary,
                "match_score": round(q.match_score, 3),
                "overlap_terms": q.overlap_terms,
                "agenda_item_id": q.agenda_item_id,
            }
            for q in self.questions
        ]

    def _create_item(self, utterance_id: str, summary: str) -> AgendaItem:
        item = AgendaItem(
            id=utterance_id,
            topic=summary,
            first_utterance_id=utterance_id,
            summaries=[summary],
        )
        self.agenda_items.append(item)
        return item

    def _open_question(self, parent: AgendaItem, utterance: Dict, summary: str) -> None:
        q = QuestionState(
            question_id=f"q{len(self.questions) + 1:03d}",
            utterance_id=utterance["id"],
            speaker=utterance["speaker"],
            summary=summary,
            agenda_item_id=parent.id,
        )
        self.questions.append(q)
        parent.questions.append(summary)
        parent.question_ids.append(q.question_id)

    def _try_answer_questions(self, parent: AgendaItem, utterance: Dict, summary: str) -> None:
        for question in self.questions:
            if question.status != "open":
                continue
            if question.agenda_item_id != parent.id:
                continue
            if question.speaker == utterance["speaker"]:
                continue

            score, overlap = self._similarity(question.summary, summary)
            if score >= self.answer_threshold:
                question.status = "answered"
                question.answered_by_utterance_id = utterance["id"]
                question.answered_by_summary = summary
                question.match_score = score
                question.overlap_terms = sorted(overlap)

    def _find_parent(self, summary: str) -> Optional[AgendaItem]:
        best_item = None
        best_score = 0.0
        for item in self.agenda_items:
            score, _ = self._similarity(summary, item.topic)
            if score > best_score:
                best_score = score
                best_item = item
        if best_item and best_score >= self.link_threshold:
            return best_item
        return None

    @staticmethod
    def _normalize_token(word: str) -> str:
        irregular = {
            "exercising": "exercise",
            "exercises": "exercise",
            "medications": "medication",
            "pills": "pill",
        }
        if word in irregular:
            return irregular[word]
        if len(word) > 5 and word.endswith("ing"):
            return word[:-3]
        if len(word) > 4 and word.endswith("s"):
            return word[:-1]
        return word

    @classmethod
    def _tokens(cls, text: str) -> Set[str]:
        cleaned = []
        for ch in text.lower():
            cleaned.append(ch if ch.isalnum() else " ")
        words = "".join(cleaned).split()
        return {
            cls._normalize_token(w)
            for w in words
            if len(w) > 2 and w not in STOPWORDS
        }

    @classmethod
    def _similarity(cls, left: str, right: str) -> tuple[float, Set[str]]:
        a = cls._tokens(left)
        b = cls._tokens(right)
        if not a or not b:
            return 0.0, set()
        overlap = a & b
        score = len(overlap) / min(len(a), len(b))
        return score, overlap

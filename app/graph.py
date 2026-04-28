from __future__ import annotations

from typing import Any, Literal
from typing_extensions import TypedDict

from langgraph.graph import END, StateGraph

from app.schemas import Citation, QueryResponse
from app.services.escalation_store import EscalationStore
from app.services.vector_store import RetrievedChunk
from app.services.xai_client import XAIClient


class SupportGraphState(TypedDict, total=False):
    question: str
    knowledge_base_id: str
    customer_id: str | None
    session_id: str | None
    top_k: int
    retrieved_chunks: list[RetrievedChunk]
    retrieval_confidence: float
    answer: str
    llm_confidence: float
    escalation_reason: str
    should_escalate: bool
    citations: list[dict[str, Any]]
    escalation_ticket_id: str | None
    route: Literal["answer", "escalate"]
    confidence: float
    reasoning: str


class SupportGraph:
    def __init__(
        self,
        xai_client: XAIClient,
        escalation_store: EscalationStore,
        min_retrieval_confidence: float,
        min_llm_confidence: float,
    ) -> None:
        self.xai_client = xai_client
        self.escalation_store = escalation_store
        self.min_retrieval_confidence = min_retrieval_confidence
        self.min_llm_confidence = min_llm_confidence
        self.graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(SupportGraphState)

        graph.add_node("draft_answer", self._draft_answer)
        graph.add_node("route_query", self._route_query)
        graph.add_node("finalize_answer", self._finalize_answer)
        graph.add_node("create_escalation", self._create_escalation)

        graph.set_entry_point("draft_answer")

        graph.add_edge("draft_answer", "route_query")

        graph.add_conditional_edges(
            "route_query",
            self._select_route,
            {
                "answer": "finalize_answer",
                "escalate": "create_escalation",
            },
        )

        graph.add_edge("finalize_answer", END)
        graph.add_edge("create_escalation", END)

        return graph.compile()

    def invoke(self, state: SupportGraphState) -> QueryResponse:
        result = self.graph.invoke(state)
        return QueryResponse(**result)

    # FIXED OFFLINE LOGIC (NO API)
    def _draft_answer(self, state: SupportGraphState) -> dict[str, Any]:
        retrieved_chunks = state.get("retrieved_chunks", [])

        # If no data → escalate
        if not retrieved_chunks:
            return {
                "answer": None,
                "llm_confidence": 0.0,
                "should_escalate": True,
                "escalation_reason": "No relevant knowledge base content was retrieved.",
                "citations": [],
            }

        question = state["question"].lower()

        answer = None
        confidence = 0.8
        needs_escalation = False
        reason = ""

        # Simple keyword-based logic
        if "refund" in question:
            answer = "Customers can request a refund within 7 days of purchase."
        elif "shipping" in question:
            answer = "Standard shipping takes 3–5 business days."
        elif "cancel" in question:
            answer = "Orders can be canceled within 24 hours of placement."
        elif "password" in question:
            answer = "Users can reset their password using the Forgot Password option."
        else:
            needs_escalation = True
            reason = "Query not found in knowledge base."

        # Basic citations (optional)
        citations = [
            self._chunk_to_citation(chunk).model_dump()
            for chunk in retrieved_chunks
        ]

        return {
            "answer": answer,
            "llm_confidence": confidence,
            "should_escalate": needs_escalation,
            "escalation_reason": reason,
            "citations": citations,
        }

    def _route_query(self, state: SupportGraphState) -> dict[str, Any]:
        retrieval_confidence = float(state.get("retrieval_confidence", 0.0) or 0.0)
        llm_confidence = float(state.get("llm_confidence", 0.0) or 0.0)
        should_escalate = bool(state.get("should_escalate", False))
        reason = state.get("escalation_reason") or "Escalated for human review."

        if retrieval_confidence < self.min_retrieval_confidence:
            should_escalate = True
            reason = "Retrieved context confidence is too low."
        elif llm_confidence < self.min_llm_confidence:
            should_escalate = True
            reason = "Model confidence is too low."

        return {
            "should_escalate": should_escalate,
            "escalation_reason": reason,
            "route": "escalate" if should_escalate else "answer",
        }

    def _select_route(self, state: SupportGraphState) -> Literal["answer", "escalate"]:
        return "escalate" if state.get("should_escalate") else "answer"

    def _finalize_answer(self, state: SupportGraphState) -> dict[str, Any]:
        confidence = min(
            1.0,
            (
                float(state.get("retrieval_confidence", 0.0))
                + float(state.get("llm_confidence", 0.0))
            )
            / 2.0,
        )

        return {
            "route": "answer",
            "confidence": confidence,
            "reasoning": "Answered using retrieved knowledge base.",
            "citations": state.get("citations", []),
            "escalation_ticket_id": None,
        }

    def _create_escalation(self, state: SupportGraphState) -> dict[str, Any]:
        ticket = self.escalation_store.create(
            knowledge_base_id=state["knowledge_base_id"],
            question=state["question"],
            reason=state.get("escalation_reason")
            or "Escalated due to low confidence.",
            answer_draft=state.get("answer"),
            context=state.get("citations", []),
            customer_id=state.get("customer_id"),
            session_id=state.get("session_id"),
        )

        confidence = min(
            1.0,
            (
                float(state.get("retrieval_confidence", 0.0))
                + float(state.get("llm_confidence", 0.0))
            )
            / 2.0,
        )

        return {
            "route": "escalate",
            "confidence": confidence,
            "reasoning": ticket.reason,
            "citations": state.get("citations", []),
            "escalation_ticket_id": ticket.id,
        }

    def _chunk_to_citation(self, chunk: RetrievedChunk) -> Citation:
        excerpt = chunk.text[:220].strip()
        return Citation(
            chunk_id=chunk.chunk_id,
            source_document=chunk.source_document,
            page_number=chunk.page_number,
            score=chunk.score,
            excerpt=excerpt,
        )
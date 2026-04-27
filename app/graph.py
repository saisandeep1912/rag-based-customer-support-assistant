from __future__ import annotations

import json
import logging
from typing import Any, Literal
from typing_extensions import TypedDict

from langgraph.graph import END, StateGraph

from app.schemas import Citation, QueryResponse
from app.services.escalation_store import EscalationStore
from app.services.vector_store import RetrievedChunk
from app.services.xai_client import XAIClient

# logger = logging.getLogger(__name__)


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

    def _draft_answer(self, state: SupportGraphState) -> dict[str, Any]:
        retrieved_chunks = state.get("retrieved_chunks", [])

        if not retrieved_chunks:
            return {
                "answer": None,
                "llm_confidence": 0.0,
                "should_escalate": True,
                "escalation_reason": "No relevant knowledge base content was retrieved.",
                "citations": [],
            }

        context = self._format_context(retrieved_chunks)
        prompt = f"""
You are a customer support assistant working only from the provided knowledge base.

Question:
{state["question"]}

Knowledge base context:
{context}

Return valid JSON only with this schema:
{{
  "answer": "A concise support answer grounded in the context.",
  "confidence": 0.0,
  "needs_escalation": false,
  "escalation_reason": "Short reason",
  "cited_chunk_ids": ["chunk-id-1"]
}}

Rules:
- If the context is insufficient, set needs_escalation to true.
- Do not invent policy details that are not in the context.
- Keep confidence between 0 and 1.
- Cite only chunk ids that appear in the context.
""".strip()

        raw = self.xai_client.generate(
            system_prompt="You answer customer support questions and emit strict JSON.",
            user_prompt=prompt,
        )
        parsed = self._parse_json(raw)
        cited_chunk_ids = set(parsed.get("cited_chunk_ids") or [])
        citations = [
            self._chunk_to_citation(chunk).model_dump()
            for chunk in retrieved_chunks
            if chunk.chunk_id in cited_chunk_ids
        ]
        if not citations:
            citations = [self._chunk_to_citation(chunk).model_dump() for chunk in retrieved_chunks]


        return {
            "answer": parsed.get("answer"),
            "llm_confidence": float(parsed.get("confidence", 0.0) or 0.0),
            "should_escalate": bool(parsed.get("needs_escalation", False)),
            "escalation_reason": parsed.get("escalation_reason")
            or "The model marked this question for human review.",
            "citations": citations,
        }

    def _route_query(self, state: SupportGraphState) -> dict[str, Any]:
        retrieval_confidence = float(state.get("retrieval_confidence", 0.0) or 0.0)
        llm_confidence = float(state.get("llm_confidence", 0.0) or 0.0)
        should_escalate = bool(state.get("should_escalate", False))
        reason = state.get("escalation_reason") or "Escalated for human review."

        if retrieval_confidence < self.min_retrieval_confidence:
            should_escalate = True
            reason = "Retrieved context confidence is too low for an automated answer."
        elif llm_confidence < self.min_llm_confidence:
            should_escalate = True
            reason = "The model confidence is too low for an automated answer."


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
            "reasoning": "Answered automatically using the retrieved knowledge base context.",
            "citations": state.get("citations", []),
            "escalation_ticket_id": None,
        }

    def _create_escalation(self, state: SupportGraphState) -> dict[str, Any]:
        ticket = self.escalation_store.create(
            knowledge_base_id=state["knowledge_base_id"],
            question=state["question"],
            reason=state.get("escalation_reason")
            or "Escalated because the system could not answer confidently.",
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

    def _format_context(self, chunks: list[RetrievedChunk]) -> str:
        return "\n\n".join(
            (
                f"Chunk ID: {chunk.chunk_id}\n"
                f"Source: {chunk.source_document}\n"
                f"Page: {chunk.page_number}\n"
                f"Score: {chunk.score:.3f}\n"
                f"Content: {chunk.text}"
            )
            for chunk in chunks
        )

    def _parse_json(self, raw: str) -> dict[str, Any]:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        return json.loads(cleaned)

    def _chunk_to_citation(self, chunk: RetrievedChunk) -> Citation:
        excerpt = chunk.text[:220].strip()
        return Citation(
            chunk_id=chunk.chunk_id,
            source_document=chunk.source_document,
            page_number=chunk.page_number,
            score=chunk.score,
            excerpt=excerpt,
        )

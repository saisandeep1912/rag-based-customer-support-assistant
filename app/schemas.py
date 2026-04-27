from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: Literal["ok"]
    service: str
    version: str


class IngestionResponse(BaseModel):
    knowledge_base_id: str
    filename: str
    chunks_indexed: int
    pages_processed: int


class QueryRequest(BaseModel):
    question: str = Field(min_length=3)
    knowledge_base_id: str = Field(default="default")
    top_k: int | None = Field(default=None, ge=1, le=10)
    customer_id: str | None = None
    session_id: str | None = None


class Citation(BaseModel):
    chunk_id: str
    source_document: str
    page_number: int | None = None
    score: float
    excerpt: str


class QueryResponse(BaseModel):
    route: Literal["answer", "escalate"]
    answer: str | None = None
    confidence: float
    reasoning: str
    citations: list[Citation] = Field(default_factory=list)
    escalation_ticket_id: str | None = None


class EscalationTicket(BaseModel):
    id: str
    knowledge_base_id: str
    question: str
    reason: str
    status: Literal["open", "resolved"]
    answer_draft: str | None = None
    customer_id: str | None = None
    session_id: str | None = None
    created_at: datetime
    updated_at: datetime
    context: list[dict[str, Any]] = Field(default_factory=list)
    human_response: str | None = None


class ResolveEscalationRequest(BaseModel):
    human_response: str = Field(min_length=3)


class KnowledgeBaseSummary(BaseModel):
    knowledge_base_id: str
    total_chunks: int

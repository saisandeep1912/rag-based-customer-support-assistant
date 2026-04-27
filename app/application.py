from __future__ import annotations

import logging
import shutil
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile

from app.config import settings
from app.graph import SupportGraph
from app.schemas import (
    HealthResponse,
    IngestionResponse,
    KnowledgeBaseSummary,
    QueryRequest,
    QueryResponse,
    ResolveEscalationRequest,
    EscalationTicket,
)
from app.services.chunking import TextChunker
from app.services.embedding_service import EmbeddingService
from app.services.escalation_store import EscalationStore
from app.services.pdf_service import PDFService
from app.services.vector_store import VectorStoreService
from app.services.xai_client import XAIClient, XAIClientError

# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
# )
# logger = logging.getLogger(__name__)


class ServiceContainer:
    def __init__(self) -> None:
        settings.ensure_directories()
        self.pdf_service = PDFService()
        self.chunker = TextChunker(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        self.embedding_service = EmbeddingService(
            model_name=settings.embedding_model_name,
            batch_size=settings.embedding_batch_size,
            hf_token=settings.hf_token,
        )
        self.vector_store = VectorStoreService(str(settings.chroma_dir))
        self.escalation_store = EscalationStore(settings.escalation_db_path)
        self.xai_client = XAIClient(
            api_key=settings.xai_api_key,
            base_url=settings.xai_base_url,
            model=settings.xai_model,
            temperature=settings.xai_temperature,
            max_output_tokens=settings.xai_max_output_tokens,
        )
        self.graph = SupportGraph(
            xai_client=self.xai_client,
            escalation_store=self.escalation_store,
            min_retrieval_confidence=settings.min_retrieval_confidence,
            min_llm_confidence=settings.min_llm_confidence,
        )

    def save_upload(self, upload: UploadFile) -> Path:
        filename = upload.filename or "knowledge-base.pdf"
        destination = settings.uploads_dir / filename
        with destination.open("wb") as file_handle:
            shutil.copyfileobj(upload.file, file_handle)
        return destination


container = ServiceContainer()
app = FastAPI(title=settings.app_name, version=settings.app_version)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        service=settings.app_name,
        version=settings.app_version,
    )


@app.post(f"{settings.api_prefix}/knowledge-bases/ingest", response_model=IngestionResponse)
def ingest_pdf(file: UploadFile = File(...)) -> IngestionResponse:
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF uploads are supported.")

    knowledge_base_id = "default"
    pdf_path = container.save_upload(file)
    pages = container.pdf_service.extract_pages(pdf_path)
    chunks = container.chunker.chunk_pages(pages)
    embeddings = container.embedding_service.embed_documents([chunk.text for chunk in chunks])
    indexed_count = container.vector_store.add_document(
        knowledge_base_id=knowledge_base_id,
        filename=pdf_path.name,
        chunks=chunks,
        embeddings=embeddings,
    )

    return IngestionResponse(
        knowledge_base_id=knowledge_base_id,
        filename=pdf_path.name,
        chunks_indexed=indexed_count,
        pages_processed=len(pages),
    )


@app.get(
    f"{settings.api_prefix}/knowledge-bases/{{knowledge_base_id}}",
    response_model=KnowledgeBaseSummary,
)
def knowledge_base_summary(knowledge_base_id: str) -> KnowledgeBaseSummary:
    return KnowledgeBaseSummary(
        knowledge_base_id=knowledge_base_id,
        total_chunks=container.vector_store.count_chunks(knowledge_base_id),
    )


def run_support_query(question: str) -> QueryResponse:
    request = QueryRequest(question=question, knowledge_base_id="default")
    top_k = request.top_k or settings.retrieval_top_k
    retrieved_chunks = container.vector_store.query(
        knowledge_base_id=request.knowledge_base_id,
        query_embedding=container.embedding_service.embed_query(request.question),
        top_k=top_k,
    )
    retrieval_confidence = max((chunk.score for chunk in retrieved_chunks), default=0.0)


    try:
        response = container.graph.invoke(
            {
                "question": request.question,
                "knowledge_base_id": request.knowledge_base_id,
                "customer_id": request.customer_id,
                "session_id": request.session_id,
                "top_k": top_k,
                "retrieved_chunks": retrieved_chunks,
                "retrieval_confidence": retrieval_confidence,
            }
        )
        return response
    except XAIClientError as exc:
        raise RuntimeError(str(exc)) from exc
    except ValueError as exc:
        raise RuntimeError(f"Invalid model response: {exc}") from exc


@app.post(f"{settings.api_prefix}/support/query", response_model=QueryResponse)
def query_support(request: QueryRequest) -> QueryResponse:
    try:
        return run_support_query(request.question)
    except RuntimeError as exc:
        message = str(exc)
        if message.startswith("xAI API error") or message.startswith("Unable to reach xAI API"):
            raise HTTPException(status_code=502, detail=message) from exc
        raise HTTPException(status_code=500, detail=message) from exc


@app.get(f"{settings.api_prefix}/escalations", response_model=list[EscalationTicket])
def list_escalations() -> list[EscalationTicket]:
    return container.escalation_store.list()


@app.get(f"{settings.api_prefix}/escalations/{{ticket_id}}", response_model=EscalationTicket)
def get_escalation(ticket_id: str) -> EscalationTicket:
    try:
        return container.escalation_store.get(ticket_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Escalation ticket not found.") from exc


@app.post(
    f"{settings.api_prefix}/escalations/{{ticket_id}}/resolve",
    response_model=EscalationTicket,
)
def resolve_escalation(
    ticket_id: str,
    request: ResolveEscalationRequest,
) -> EscalationTicket:
    try:
        return container.escalation_store.resolve(ticket_id, request.human_response)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Escalation ticket not found.") from exc

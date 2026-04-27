from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import chromadb

from app.services.chunking import TextChunk


@dataclass(slots=True)
class RetrievedChunk:
    chunk_id: str
    text: str
    source_document: str
    page_number: int | None
    score: float
    metadata: dict[str, Any]


class VectorStoreService:
    def __init__(self, persist_directory: str) -> None:
        self.client = chromadb.PersistentClient(path=persist_directory)

    def _collection_name(self, knowledge_base_id: str) -> str:
        sanitized = re.sub(r"[^a-zA-Z0-9_-]+", "-", knowledge_base_id).strip("-")
        if not sanitized:
            sanitized = "default"
        return sanitized[:63]

    def _get_collection(self, knowledge_base_id: str):
        return self.client.get_or_create_collection(
            name=self._collection_name(knowledge_base_id),
            metadata={"hnsw:space": "cosine"},
        )

    def add_document(
        self,
        knowledge_base_id: str,
        filename: str,
        chunks: list[TextChunk],
        embeddings: list[list[float]],
    ) -> int:
        if not chunks:
            return 0

        collection = self._get_collection(knowledge_base_id)
        ids = [
            f"{self._collection_name(knowledge_base_id)}-{filename}-{chunk.chunk_index}"
            for chunk in chunks
        ]
        metadatas = [
            {
                "source_document": filename,
                "page_number": chunk.page_number,
                "chunk_index": chunk.chunk_index,
            }
            for chunk in chunks
        ]
        documents = [chunk.text for chunk in chunks]

        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        return len(ids)

    def query(
        self,
        knowledge_base_id: str,
        query_embedding: list[float],
        top_k: int,
    ) -> list[RetrievedChunk]:
        collection = self._get_collection(knowledge_base_id)
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        ids = results.get("ids", [[]])[0]

        retrieved: list[RetrievedChunk] = []
        for chunk_id, text, metadata, distance in zip(ids, documents, metadatas, distances):
            score = max(0.0, min(1.0, 1.0 - float(distance)))
            retrieved.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    text=text,
                    source_document=str(metadata.get("source_document", "unknown")),
                    page_number=metadata.get("page_number"),
                    score=score,
                    metadata=metadata,
                )
            )
        return retrieved

    def count_chunks(self, knowledge_base_id: str) -> int:
        collection = self._get_collection(knowledge_base_id)
        return collection.count()

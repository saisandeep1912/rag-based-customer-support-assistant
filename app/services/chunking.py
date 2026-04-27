from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class TextChunk:
    text: str
    page_number: int
    chunk_index: int


class TextChunker:
    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_pages(self, pages: list[tuple[int, str]]) -> list[TextChunk]:
        chunks: list[TextChunk] = []
        chunk_index = 0

        for page_number, page_text in pages:
            normalized = " ".join(page_text.split())
            if not normalized:
                continue

            start = 0
            while start < len(normalized):
                end = min(len(normalized), start + self.chunk_size)
                chunk_text = normalized[start:end].strip()
                if chunk_text:
                    chunks.append(
                        TextChunk(
                            text=chunk_text,
                            page_number=page_number,
                            chunk_index=chunk_index,
                        )
                    )
                    chunk_index += 1
                if end >= len(normalized):
                    break
                start = max(0, end - self.chunk_overlap)

        return chunks

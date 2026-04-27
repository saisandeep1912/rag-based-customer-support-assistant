from __future__ import annotations

from functools import lru_cache

from sentence_transformers import SentenceTransformer
from transformers.utils import logging as transformers_logging


class EmbeddingService:
    def __init__(
        self,
        model_name: str,
        batch_size: int = 32,
        hf_token: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.hf_token = hf_token

    @lru_cache(maxsize=1)
    def _model(self) -> SentenceTransformer:
        previous_verbosity = transformers_logging.get_verbosity()
        transformers_logging.set_verbosity_error()
        try:
            return SentenceTransformer(self.model_name, token=self.hf_token)
        finally:
            transformers_logging.set_verbosity(previous_verbosity)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        embeddings = self._model().encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

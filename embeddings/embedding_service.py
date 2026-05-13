import logging
from typing import List

from openai import AzureOpenAI

from config import Settings

logger = logging.getLogger(__name__)


class AzureEmbeddingService:
    def __init__(self, settings: Settings) -> None:
        self._client = AzureOpenAI(
            api_key=settings.azure_openai_api_key,
            azure_endpoint=settings.azure_openai_endpoint,
            api_version=settings.azure_openai_api_version,
        )
        self._deployment = settings.azure_openai_embedding_deployment

    def embed_text(self, text: str) -> List[float]:
        if not text.strip():
            raise ValueError("Cannot embed empty text")

        try:
            response = self._client.embeddings.create(
                model=self._deployment,
                input=text,
            )
        except Exception as exc:
            logger.exception("Embedding request failed")
            raise RuntimeError(f"Embedding failed: {exc}") from exc

        return response.data[0].embedding

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        clean_texts = [t for t in texts if t and t.strip()]
        if not clean_texts:
            raise ValueError("Cannot embed an empty text batch")

        try:
            response = self._client.embeddings.create(
                model=self._deployment,
                input=clean_texts,
            )
        except Exception as exc:
            logger.exception("Batch embedding request failed")
            raise RuntimeError(f"Batch embedding failed: {exc}") from exc

        return [item.embedding for item in response.data]

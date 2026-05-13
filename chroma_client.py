import logging
from typing import Dict, List, Optional

import chromadb

from config import Settings
from ingestion.types import PageUnit

logger = logging.getLogger(__name__)


class ChromaVectorStore:
    def __init__(self, settings: Settings) -> None:
        client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
        self._collection = client.get_or_create_collection(name=settings.chroma_collection_name)

    def upsert_units(self, units: List[PageUnit], embeddings: List[List[float]]) -> None:
        if len(units) != len(embeddings):
            raise ValueError("Units count does not match embeddings count")
        if not units:
            return

        self._collection.upsert(
            ids=[u.id for u in units],
            embeddings=embeddings,
            documents=[u.content for u in units],
            metadatas=[u.metadata() for u in units],
        )
        logger.info("Upserted %s page-units into ChromaDB", len(units))

    def query(
        self,
        query_embedding: List[float],
        top_k: int = 3,
        metadata_filter: Optional[Dict[str, object]] = None,
    ) -> List[Dict[str, object]]:
        result = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=metadata_filter,
        )

        ids = result.get("ids", [[]])[0]
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        matches: List[Dict[str, object]] = []
        for i in range(len(ids)):
            matches.append(
                {
                    "id": ids[i],
                    "content": documents[i],
                    "metadata": metadatas[i],
                    "distance": distances[i] if i < len(distances) else None,
                }
            )

        logger.info("Retrieved %s results from ChromaDB", len(matches))
        return matches

import re
from typing import Dict, List, Optional

from embeddings.embedding_service import AzureEmbeddingService
from vectorstore.chroma_client import ChromaVectorStore


class RetrievalTool:
    def __init__(self, embedder: AzureEmbeddingService, store: ChromaVectorStore) -> None:
        self._embedder = embedder
        self._store = store

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        metadata_filter: Optional[Dict[str, object]] = None,
    ) -> List[Dict[str, object]]:
        query_embedding = self._embedder.embed_text(query)
        return self._store.query(
            query_embedding=query_embedding,
            top_k=top_k,
            metadata_filter=metadata_filter,
        )


def extract_relevant_paragraphs(
    query: str,
    retrieved_units: List[Dict[str, object]],
    max_paragraphs_per_unit: int = 3,
) -> List[Dict[str, object]]:
    query_terms = set(re.findall(r"\w+", query.lower()))
    focused_units: List[Dict[str, object]] = []

    for unit in retrieved_units:
        content = str(unit.get("content", ""))
        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]

        if not paragraphs:
            focused_units.append(unit)
            continue

        scored = []
        for p in paragraphs:
            tokens = set(re.findall(r"\w+", p.lower()))
            overlap = len(query_terms.intersection(tokens))
            scored.append((overlap, p))

        scored.sort(key=lambda item: item[0], reverse=True)
        chosen = [p for score, p in scored[:max_paragraphs_per_unit] if score > 0]

        focused_units.append(
            {
                **unit,
                "content": "\n\n".join(chosen) if chosen else content,
            }
        )

    return focused_units

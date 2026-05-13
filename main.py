import argparse
import asyncio
import logging
from pathlib import Path
from typing import List, Optional

from agent.agent import SelfReflectiveRAGAgent
from agent.tools import RetrievalTool
from config import Settings
from embeddings.embedding_service import AzureEmbeddingService
from ingestion.loader import DocumentLoader
from ingestion.types import PageUnit
from vectorstore.chroma_client import ChromaVectorStore


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def ingest_documents(
    files: List[str],
    loader: DocumentLoader,
    embedder: AzureEmbeddingService,
    store: ChromaVectorStore,
) -> int:
    all_units: List[PageUnit] = []

    for file in files:
        units = loader.load(file)
        if not units:
            logging.warning("No units extracted from file: %s", file)
            continue
        all_units.extend(units)
        logging.info("Extracted %s units from %s", len(units), file)

    if not all_units:
        raise ValueError("No content extracted from any input document")

    embeddings = embedder.embed_texts([u.content for u in all_units])
    store.upsert_units(all_units, embeddings)

    return len(all_units)


async def run_query(
    question: str,
    settings: Settings,
    embedder: AzureEmbeddingService,
    store: ChromaVectorStore,
    source_filter: Optional[str],
) -> None:
    retrieval_tool = RetrievalTool(embedder=embedder, store=store)
    agent = SelfReflectiveRAGAgent(settings=settings, retrieval_tool=retrieval_tool)

    metadata_filter = {"source": source_filter} if source_filter else None
    result = await agent.ask(question=question, metadata_filter=metadata_filter)

    print("\n=== Final Answer ===\n")
    print(result["answer"])
    print("\n=== Reflection Trace ===\n")
    for item in result["iterations"]:
        print(
            f"Round {item['round']} | Query: {item['query']} | "
            f"Retrieved: {item['retrieved_count']} | "
            f"Sufficient: {item['reflection']['is_sufficient']}"
        )
        if item["reflection"].get("missing_points"):
            print(f"Missing: {item['reflection']['missing_points']}")
        print("-")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Agentic Self-Reflective RAG (page-level chunks)")
    parser.add_argument(
        "--ingest",
        nargs="*",
        default=[],
        help="Document paths to ingest (pdf/docx/pptx/xlsx/image/html)",
    )
    parser.add_argument("--query", type=str, default="", help="Question to ask")
    parser.add_argument(
        "--source-filter",
        type=str,
        default="",
        help="Optional source metadata filter (exact document name)",
    )
    return parser


def validate_input_files(paths: List[str]) -> List[str]:
    valid = []
    for p in paths:
        path = Path(p)
        if path.exists() and path.is_file():
            valid.append(str(path))
        else:
            logging.warning("Skipping invalid file path: %s", p)
    return valid


def main() -> None:
    setup_logging()
    parser = build_parser()
    args = parser.parse_args()

    settings = Settings.from_env()

    loader = DocumentLoader()
    embedder = AzureEmbeddingService(settings)
    store = ChromaVectorStore(settings)

    ingest_paths = validate_input_files(args.ingest)
    if ingest_paths:
        count = ingest_documents(ingest_paths, loader, embedder, store)
        logging.info("Ingestion complete. Total page-equivalent units indexed: %s", count)

    if args.query.strip():
        asyncio.run(
            run_query(
                question=args.query,
                settings=settings,
                embedder=embedder,
                store=store,
                source_filter=args.source_filter.strip() or None,
            )
        )
    else:
        logging.info("No query provided. Use --query to run retrieval and agent loop.")


if __name__ == "__main__":
    main()

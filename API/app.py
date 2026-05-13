import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from agent.agent import SelfReflectiveRAGAgent
from agent.tools import RetrievalTool
from config import Settings
from embeddings.embedding_service import AzureEmbeddingService
from ingestion.loader import DocumentLoader
from ingestion.types import PageUnit
from vectorstore.chroma_client import ChromaVectorStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Agentic Self-Reflective RAG API", version="1.0.0")


class IngestRequest(BaseModel):
    file_paths: List[str] = Field(default_factory=list)


class QueryRequest(BaseModel):
    question: str
    source_filter: Optional[str] = None
    use_paragraph_focus: bool = True


class QueryResponse(BaseModel):
    answer: str
    answer_mode: str
    iterations: List[Dict[str, Any]]
    final_query: str


def _get_runtime() -> Dict[str, Any]:
    runtime = getattr(app.state, "runtime", None)
    if runtime is None:
        raise RuntimeError("API runtime is not initialized")
    return runtime


@app.on_event("startup")
def startup() -> None:
    try:
        settings = Settings.from_env()
    except Exception as exc:
        logger.exception("Failed to initialize settings")
        raise RuntimeError(f"Failed to initialize settings: {exc}") from exc

    embedder = AzureEmbeddingService(settings)
    store = ChromaVectorStore(settings)
    loader = DocumentLoader()

    retrieval_tool = RetrievalTool(embedder=embedder, store=store)
    agent = SelfReflectiveRAGAgent(settings=settings, retrieval_tool=retrieval_tool)

    app.state.runtime = {
        "settings": settings,
        "embedder": embedder,
        "store": store,
        "loader": loader,
        "agent": agent,
    }
    logger.info("API runtime initialized")


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/ingest")
def ingest(req: IngestRequest) -> Dict[str, Any]:
    runtime = _get_runtime()
    loader: DocumentLoader = runtime["loader"]
    embedder: AzureEmbeddingService = runtime["embedder"]
    store: ChromaVectorStore = runtime["store"]

    valid_paths: List[Path] = []
    for file_path in req.file_paths:
        p = Path(file_path)
        if p.exists() and p.is_file():
            valid_paths.append(p)
        else:
            logger.warning("Skipping invalid path: %s", file_path)

    if not valid_paths:
        raise HTTPException(status_code=400, detail="No valid input files found")

    all_units: List[PageUnit] = []
    failed_files: List[Dict[str, str]] = []

    for p in valid_paths:
        try:
            units = loader.load(str(p))
            if units:
                all_units.extend(units)
            else:
                failed_files.append({"file": str(p), "error": "No extractable content"})
        except Exception as exc:
            logger.exception("Failed to ingest file: %s", p)
            failed_files.append({"file": str(p), "error": str(exc)})

    if not all_units:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "No content extracted from provided files",
                "failures": failed_files,
            },
        )

    try:
        embeddings = embedder.embed_texts([u.content for u in all_units])
        store.upsert_units(all_units, embeddings)
    except Exception as exc:
        logger.exception("Embedding or vector upsert failed")
        raise HTTPException(status_code=500, detail=f"Vectorization failed: {exc}") from exc

    return {
        "indexed_units": len(all_units),
        "indexed_sources": sorted({u.source for u in all_units}),
        "failed_files": failed_files,
    }


@app.post("/ingest/upload")
def ingest_upload(files: List[UploadFile] = File(...)) -> Dict[str, Any]:
    temp_paths: List[str] = []
    temp_dir: Optional[str] = None
    try:
        temp_dir = tempfile.mkdtemp(prefix="rag_upload_")
        for idx, upload in enumerate(files, start=1):
            original_name = Path(upload.filename or f"uploaded_{idx}.bin").name
            target_path = Path(temp_dir) / original_name

            # Ensure unique path if duplicate file names are uploaded in one request.
            if target_path.exists():
                stem = target_path.stem
                suffix = target_path.suffix
                target_path = Path(temp_dir) / f"{stem}_{idx}{suffix}"

            content = upload.file.read()
            target_path.write_bytes(content)
            temp_paths.append(str(target_path))

        return ingest(IngestRequest(file_paths=temp_paths))
    finally:
        for p in temp_paths:
            try:
                Path(p).unlink(missing_ok=True)
            except Exception:
                logger.warning("Failed to cleanup temp file: %s", p)
        if temp_dir:
            try:
                os.rmdir(temp_dir)
            except Exception:
                logger.warning("Failed to cleanup temp directory: %s", temp_dir)


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest) -> QueryResponse:
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    runtime = _get_runtime()
    agent: SelfReflectiveRAGAgent = runtime["agent"]

    metadata_filter = {"source": req.source_filter} if req.source_filter else None

    try:
        result = await agent.ask(
            question=req.question,
            metadata_filter=metadata_filter,
            use_paragraph_focus=req.use_paragraph_focus,
        )
    except Exception as exc:
        logger.exception("Query execution failed")
        raise HTTPException(status_code=500, detail=f"Query execution failed: {exc}") from exc

    return QueryResponse(
        answer=result["answer"],
        answer_mode=result.get("answer_mode", "document"),
        iterations=result["iterations"],
        final_query=result["final_query"],
    )

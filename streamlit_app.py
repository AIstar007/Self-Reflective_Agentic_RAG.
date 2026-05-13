import asyncio
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

from agent.agent import SelfReflectiveRAGAgent
from agent.tools import RetrievalTool
from config import Settings
from embeddings.embedding_service import AzureEmbeddingService
from ingestion.loader import DocumentLoader
from ingestion.types import PageUnit
from vectorstore.chroma_client import ChromaVectorStore


@st.cache_resource
def build_runtime() -> Dict[str, Any]:
    settings = Settings.from_env()
    loader = DocumentLoader()
    embedder = AzureEmbeddingService(settings)
    store = ChromaVectorStore(settings)
    retrieval_tool = RetrievalTool(embedder=embedder, store=store)
    agent = SelfReflectiveRAGAgent(settings=settings, retrieval_tool=retrieval_tool)
    return {
        "settings": settings,
        "loader": loader,
        "embedder": embedder,
        "store": store,
        "agent": agent,
    }


def ingest_uploaded_files(runtime: Dict[str, Any], uploads: List[Any]) -> Dict[str, Any]:
    loader: DocumentLoader = runtime["loader"]
    embedder: AzureEmbeddingService = runtime["embedder"]
    store: ChromaVectorStore = runtime["store"]

    all_units: List[PageUnit] = []
    failed_files: List[Dict[str, str]] = []

    with tempfile.TemporaryDirectory(prefix="streamlit_rag_upload_") as temp_dir:
        for upload in uploads:
            temp_path = Path(temp_dir) / Path(upload.name).name
            temp_path.write_bytes(upload.getbuffer())
            try:
                units = loader.load(str(temp_path))
                if units:
                    all_units.extend(units)
                else:
                    failed_files.append({"file": upload.name, "error": "No extractable content"})
            except Exception as exc:
                failed_files.append({"file": upload.name, "error": str(exc)})

    if not all_units:
        return {
            "indexed_units": 0,
            "indexed_sources": [],
            "failed_files": failed_files,
        }

    embeddings = embedder.embed_texts([u.content for u in all_units])
    store.upsert_units(all_units, embeddings)

    return {
        "indexed_units": len(all_units),
        "indexed_sources": sorted({u.source for u in all_units}),
        "failed_files": failed_files,
    }


async def run_question(
    runtime: Dict[str, Any],
    question: str,
    source_filter: Optional[str],
    use_paragraph_focus: bool,
) -> Dict[str, Any]:
    agent: SelfReflectiveRAGAgent = runtime["agent"]
    metadata_filter = {"source": source_filter} if source_filter else None
    return await agent.ask(
        question=question,
        metadata_filter=metadata_filter,
        use_paragraph_focus=use_paragraph_focus,
    )


def main() -> None:
    st.set_page_config(page_title="Self-Reflective Agentic RAG", page_icon="✈️", layout="wide")
    st.title("Self-Reflective Agentic RAG UI")
    st.caption("Ingest policy documents, run queries, and inspect reflection rounds.")

    try:
        runtime = build_runtime()
    except Exception as exc:
        st.error(f"Failed to initialize runtime. Check your .env configuration. Error: {exc}")
        return

    left, right = st.columns([1, 2])

    with left:
        st.subheader("Ingest Documents")
        uploads = st.file_uploader(
            "Upload files",
            accept_multiple_files=True,
            type=["pdf", "docx", "pptx", "xlsx", "png", "jpg", "jpeg", "html", "htm"],
        )
        if st.button("Ingest Uploaded Files", use_container_width=True):
            if not uploads:
                st.warning("Please upload at least one file.")
            else:
                with st.spinner("Ingesting and indexing documents..."):
                    try:
                        summary = ingest_uploaded_files(runtime, uploads)
                        st.success(f"Indexed units: {summary['indexed_units']}")
                        if summary["indexed_sources"]:
                            st.write("Indexed sources:", summary["indexed_sources"])
                        if summary["failed_files"]:
                            st.warning("Some files failed to process:")
                            st.json(summary["failed_files"])
                    except Exception as exc:
                        st.error(f"Ingestion failed: {exc}")

    with right:
        st.subheader("Ask Questions")
        question = st.text_area("Question", placeholder="What are the baggage rules for domestic flights?")
        source_filter = st.text_input("Optional source filter (exact document name)")
        use_paragraph_focus = st.checkbox("Use paragraph focus", value=True)

        if st.button("Run Query", type="primary", use_container_width=True):
            if not question.strip():
                st.warning("Please enter a question.")
            else:
                with st.spinner("Running self-reflective query loop..."):
                    try:
                        result = asyncio.run(
                            run_question(
                                runtime=runtime,
                                question=question.strip(),
                                source_filter=source_filter.strip() or None,
                                use_paragraph_focus=use_paragraph_focus,
                            )
                        )
                    except Exception as exc:
                        st.error(f"Query failed: {exc}")
                        return

                st.success("Query complete")
                st.markdown("### Final Answer")
                st.write(result.get("answer", ""))

                meta_cols = st.columns(2)
                meta_cols[0].metric("Answer Mode", result.get("answer_mode", "unknown"))
                meta_cols[1].metric("Reflection Rounds", len(result.get("iterations", [])))

                st.markdown("### Reflection Trace")
                for item in result.get("iterations", []):
                    title = f"Round {item.get('round')} | Retrieved: {item.get('retrieved_count', 0)}"
                    with st.expander(title, expanded=False):
                        st.write("Query:", item.get("query", ""))
                        st.write("Answer:")
                        st.write(item.get("answer", ""))
                        st.write("Reflection:")
                        st.json(item.get("reflection", {}))


if __name__ == "__main__":
    main()

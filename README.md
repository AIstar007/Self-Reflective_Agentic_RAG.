# Agentic Self-Reflective RAG (Page-Level Chunks)

Production-ready Python implementation of an agentic self-reflective RAG pipeline with strict page-equivalent chunking.

## Stack

- Orchestration: Semantic Kernel (Microsoft ecosystem fallback for Microsoft Agent Framework requirement)
- LLM + Embeddings: Azure OpenAI
- Vector DB: ChromaDB
- Chunking policy: page-level or logical-unit level only (no token chunking, no text splitter)

## Supported Document Types

- PDF: page-wise extraction
- DOCX: heading/section-wise grouping
- PPTX: slide-wise extraction
- XLSX: sheet-wise grouping
- Images: OCR extraction as one page-equivalent unit
- HTML: section-wise grouping

## Project Structure

```text
rag-agent-app/
  ingestion/
    loader.py
    pdf_loader.py
    docx_loader.py
    ppt_loader.py
    excel_loader.py
    image_loader.py
    html_loader.py
    types.py
  embeddings/
    embedding_service.py
  vectorstore/
    chroma_client.py
  agent/
    agent.py
    tools.py
    reflection.py
  api/
    app.py
  main.py
  config.py
  requirements.txt
```

## Setup

1. Create and activate virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

1. Create your local .env file from template:

```powershell
Copy-Item .env.template .env
```

1. Fill the .env values:

```env
AZURE_OPENAI_API_KEY=<your_key>
AZURE_OPENAI_ENDPOINT=https://<your-resource>.openai.azure.com/
AZURE_OPENAI_CHAT_DEPLOYMENT=<chat_deployment_name>
AZURE_OPENAI_API_VERSION=2024-10-21
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-large
CHROMA_PERSIST_DIR=./chroma_db
CHROMA_COLLECTION_NAME=page_units
RETRIEVAL_TOP_K=3
RETRIEVAL_WEAK_MATCH_DISTANCE=1.2
MAX_REFLECTION_ROUNDS=3
```

The app auto-loads `.env` at startup using `python-dotenv`.

Note: For OCR, install Tesseract on your machine and ensure it is available in PATH.

## Example Ingestion

```bash
python main.py --ingest "./samples/airline_faq.pdf" "./samples/policy.docx" "./samples/routes.pptx" "./samples/tables.xlsx" "./samples/poster.png" "./samples/site.html"
```

## Example Query Execution

```bash
python main.py --query "What are the baggage rules for domestic flights?"
```

## Combined Flow

```bash
python main.py --ingest "./samples/airline_faq.pdf" --query "Summarize check-in and baggage policies"
```

## API Execution Flow

Start the API server from the rag-agent-app directory:

```bash
uvicorn api.app:app --reload
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Ingest by file paths:

```bash
curl -X POST http://127.0.0.1:8000/ingest \
  -H "Content-Type: application/json" \
  -d "{\"file_paths\": [\"./samples/airline_faq.pdf\", \"./samples/policy.docx\"]}"
```

Ingest by upload:

```bash
curl -X POST http://127.0.0.1:8000/ingest/upload \
  -F "files=@./samples/airline_faq.pdf" \
  -F "files=@./samples/site.html"
```

Run query:

```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d "{\"question\": \"What are baggage rules?\", \"use_paragraph_focus\": true}"
```

Note: `source_filter` must exactly match the indexed `source` name. For upload ingestion, source is the original uploaded filename.

## Streamlit UI

Run the UI from the `rag-agent-app` directory:

```bash
streamlit run streamlit_app.py
```

The UI supports:

- Upload and ingest supported document formats
- Ask questions with optional `source_filter`
- View final answer, answer mode, and per-round reflection trace

## Self-Reflection Loop

For each query, the agent executes:

1. Retrieve top-k page-units from ChromaDB.
2. Generate draft answer with Azure OpenAI.
3. Run reflection prompt to score sufficiency and gaps.
4. If insufficient, refine query and retrieve again.
5. Stop after sufficiency or max rounds (default: 3).

## Error Handling Included

- Missing environment variables
- Unsupported file type
- Empty extraction result
- Embedding API failures
- Empty retrieval results

## Notes

- No LangChain used.
- No token-level chunking used.
- Retrieval defaults to top 3 units to control token usage.
- Optional paragraph focusing is enabled in the agent for large page units.
- Routing behavior: strong retrieval match answers from documents; weak retrieval match answers from external LLM knowledge.
- Match strength is controlled by `RETRIEVAL_WEAK_MATCH_DISTANCE` (lower distance is stronger).
- If you ingested files before filename-preserving upload handling, re-ingest them so `source_filter` works with real document names.

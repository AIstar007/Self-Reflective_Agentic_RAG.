<div align="center">

# 🧠 Self-Reflective Agentic RAG

### Production-ready · Page-Level Chunks · Azure OpenAI · ChromaDB

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Azure OpenAI](https://img.shields.io/badge/Azure_OpenAI-LLM_%26_Embeddings-0089D6?style=for-the-badge&logo=microsoftazure&logoColor=white)](https://azure.microsoft.com/en-us/products/ai-services/openai-service)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-FF6B35?style=for-the-badge&logo=databricks&logoColor=white)](https://www.trychroma.com/)
[![Semantic Kernel](https://img.shields.io/badge/Semantic_Kernel-Orchestration-5C2D91?style=for-the-badge&logo=microsoft&logoColor=white)](https://learn.microsoft.com/en-us/semantic-kernel/overview/)
[![License: MIT](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)](LICENSE)

<br/>

> **An agentic RAG pipeline that thinks twice before it answers.**  
> Retrieves → Reflects → Refines — automatically, up to N rounds.

<br/>

[🚀 Quick Start](#-quick-start) · [📐 Architecture](#-architecture) · [📂 Project Structure](#-project-structure) · [⚙️ Configuration](#️-configuration) · [🖥️ API](#️-api-usage) · [🎨 Streamlit UI](#-streamlit-ui)

---

</div>

## ✨ What Makes This Different

| Feature | This Project | Typical RAG |
|---|---|---|
| **Chunking** | Page / Slide / Section (semantic units) | Token splitter |
| **Self-Reflection** | ✅ Multi-round gap analysis | ❌ Single pass |
| **Routing** | Strong match → docs, Weak match → LLM | Always docs |
| **LangChain** | ❌ Not used | Usually yes |
| **Document Types** | PDF, DOCX, PPTX, XLSX, Images, HTML | PDF only |

---

## 📐 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Query Entry Point                        │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Semantic Kernel Agent                       │
│                                                                 │
│   ┌──────────┐    ┌──────────────┐    ┌──────────────────────┐  │
│   │ Retrieve │───▶│ Draft Answer │───▶│  Reflection Prompt   │  │
│   │ Top-K    │    │ (Azure OAI)  │    │  Score: Sufficient?  │  │
│   └──────────┘    └──────────────┘    └──────────┬───────────┘  │
│        ▲                                         │              │
│        │          ┌───────────────┐    YES ──────▼──────────    │
│        └──────────│ Refine Query  │              RETURN         │
│                   │  + Re-Fetch   │◀── NO                       │
│                   └───────────────┘   (max 3 rounds)            │
└─────────────────────────────────────────────────────────────────┘
                             │
              ┌──────────────┴──────────────┐
              ▼                             ▼
   Strong Match (dist < 1.2)      Weak Match (dist ≥ 1.2)
   Answer from Documents          Answer from LLM Knowledge
```

---

## 🗂️ Supported Document Types

| Type | Chunking Strategy |
|---|---|
| 📄 **PDF** | One chunk per page |
| 📝 **DOCX** | One chunk per heading/section |
| 📊 **PPTX** | One chunk per slide |
| 📈 **XLSX** | One chunk per sheet |
| 🖼️ **Images** | OCR → one chunk per image |
| 🌐 **HTML** | One chunk per `<section>` / semantic block |

---

## 📂 Project Structure

```
rag-agent-app/
│
├── 📁 ingestion/
│   ├── loader.py            # Document type router
│   ├── pdf_loader.py        # Page-wise PDF extraction
│   ├── docx_loader.py       # Heading/section grouping
│   ├── ppt_loader.py        # Slide-wise extraction
│   ├── excel_loader.py      # Sheet-wise grouping
│   ├── image_loader.py      # OCR via Tesseract
│   ├── html_loader.py       # Section-wise grouping
│   └── types.py             # Shared data models
│
├── 📁 embeddings/
│   └── embedding_service.py # Azure OpenAI embeddings
│
├── 📁 vectorstore/
│   └── chroma_client.py     # ChromaDB CRUD + similarity search
│
├── 📁 agent/
│   ├── agent.py             # Orchestration + reflection loop
│   ├── tools.py             # SK tool definitions
│   └── reflection.py        # Sufficiency scoring prompt
│
├── 📁 api/
│   └── app.py               # FastAPI endpoints
│
├── streamlit_app.py         # Interactive UI
├── main.py                  # CLI entry point
├── config.py                # Env config loader
└── requirements.txt
```

---

## 🚀 Quick Start

### 1 · Clone the repo

```bash
git clone https://github.com/AIstar007/Self-Reflective_Agentic_RAG..git
cd Self-Reflective_Agentic_RAG/rag-agent-app
```

### 2 · Create virtual environment & install

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

> **OCR support:** Install [Tesseract](https://github.com/tesseract-ocr/tesseract) and ensure it's on your `PATH`.

### 3 · Configure environment

```powershell
# Windows
Copy-Item .env.template .env
```
```bash
# macOS / Linux
cp .env.template .env
```

---

## ⚙️ Configuration

Edit your `.env` file:

```env
# ── Azure OpenAI ──────────────────────────────────────────
AZURE_OPENAI_API_KEY=<your_key>
AZURE_OPENAI_ENDPOINT=https://<your-resource>.openai.azure.com/
AZURE_OPENAI_CHAT_DEPLOYMENT=<chat_deployment_name>
AZURE_OPENAI_API_VERSION=2024-10-21
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-large

# ── ChromaDB ──────────────────────────────────────────────
CHROMA_PERSIST_DIR=./chroma_db
CHROMA_COLLECTION_NAME=page_units

# ── Retrieval & Reflection ─────────────────────────────────
RETRIEVAL_TOP_K=3
RETRIEVAL_WEAK_MATCH_DISTANCE=1.2   # lower = stronger match required
MAX_REFLECTION_ROUNDS=3
```

---

## 💻 CLI Usage

### Ingest documents

```bash
python main.py --ingest \
  "./samples/airline_faq.pdf" \
  "./samples/policy.docx" \
  "./samples/routes.pptx" \
  "./samples/tables.xlsx" \
  "./samples/poster.png" \
  "./samples/site.html"
```

### Run a query

```bash
python main.py --query "What are the baggage rules for domestic flights?"
```

### Combined flow

```bash
python main.py \
  --ingest "./samples/airline_faq.pdf" \
  --query  "Summarize check-in and baggage policies"
```

---

## 🖥️ API Usage

Start the server:

```bash
uvicorn api.app:app --reload
```

### Endpoints

**Health check**
```bash
curl http://127.0.0.1:8000/health
```

**Ingest by file path**
```bash
curl -X POST http://127.0.0.1:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"file_paths": ["./samples/airline_faq.pdf", "./samples/policy.docx"]}'
```

**Ingest by upload**
```bash
curl -X POST http://127.0.0.1:8000/ingest/upload \
  -F "files=@./samples/airline_faq.pdf" \
  -F "files=@./samples/site.html"
```

**Query**
```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are baggage rules?", "use_paragraph_focus": true}'
```

> 💡 `source_filter` must exactly match the indexed `source` name. For upload ingestion, source is the original filename.

---

## 🎨 Streamlit UI

```bash
streamlit run streamlit_app.py
```

The UI provides:

- 📤 Upload & ingest any supported document format
- 💬 Ask questions with optional `source_filter`
- 🔍 View final answer, answer mode, and **per-round reflection trace**

---

## 🔄 Self-Reflection Loop (Deep Dive)

```
Round 1 ──▶ Retrieve top-k page-units from ChromaDB
        ──▶ Generate draft answer (Azure OpenAI)
        ──▶ Reflection: score sufficiency, identify gaps
              │
              ├── ✅ Sufficient? ──▶ Return answer
              │
              └── ❌ Gaps found? ──▶ Refine query ──▶ Round 2 ...
                                                          │
                                                    max 3 rounds
                                                    then return best
```

---

## 🛡️ Error Handling

| Scenario | Handled |
|---|---|
| Missing environment variables | ✅ |
| Unsupported file type | ✅ |
| Empty extraction result | ✅ |
| Embedding API failures | ✅ |
| Empty retrieval results | ✅ |

---

## 🧱 Tech Stack

| Layer | Technology |
|---|---|
| Orchestration | [Semantic Kernel](https://learn.microsoft.com/en-us/semantic-kernel/overview/) |
| LLM + Embeddings | [Azure OpenAI](https://azure.microsoft.com/en-us/products/ai-services/openai-service) |
| Vector Store | [ChromaDB](https://www.trychroma.com/) |
| API | [FastAPI](https://fastapi.tiangolo.com/) |
| UI | [Streamlit](https://streamlit.io/) |
| OCR | [Tesseract](https://github.com/tesseract-ocr/tesseract) |

---

<div align="center">

Made with ❤️ · [⭐ Star on GitHub](https://github.com/AIstar007/Self-Reflective_Agentic_RAG..git) · [🐛 Report an Issue](https://github.com/AIstar007/Self-Reflective_Agentic_RAG./issues)

</div>

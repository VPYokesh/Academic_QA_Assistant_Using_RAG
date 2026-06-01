# 🎓 Academic QA Assistant Using RAG

> A production-quality **Retrieval-Augmented Generation (RAG)** question-answering system built for academic research. Upload your study documents, ask questions in natural language, and get grounded, cited answers — complete with hallucination detection, cross-encoder re-ranking, and a polished Streamlit UI.

![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.55%2B-red?logo=streamlit)
![LangChain](https://img.shields.io/badge/LangChain-1.2%2B-green)
![ChromaDB](https://img.shields.io/badge/ChromaDB-1.5%2B-purple)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## ✨ Features

| Feature | Description |
|---|---|
| 📄 **Multi-format Ingestion** | Upload PDF, DOCX, TXT, and CSV files |
| 🔍 **Semantic Search** | `intfloat/multilingual-e5-large` bi-encoder embeddings via ChromaDB |
| 🔁 **Cross-Encoder Re-Ranking** | `ms-marco-MiniLM-L-6-v2` re-ranks top-K candidates for precision |
| 🤖 **LLM Inference (Groq)** | `llama-3.1-8b-instant` via Groq API for ultra-fast responses |
| 🛡️ **Hallucination Detection** | `all-MiniLM-L6-v2` cosine similarity scoring with HIGH / MEDIUM / LOW labels |
| 💬 **Conversational Memory** | LangGraph `MemorySaver` keeps multi-turn context per session |
| 📊 **Source Cards** | Every answer shows cited sources, page numbers, and relevance % |
| ⚙️ **Live Parameter Tuning** | Adjust Temperature, Max Tokens, and Top-K without restarting |
| 🗄️ **DB Management** | Per-document delete, stats tiles, and ingestion activity log |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Streamlit UI                          │
│  ┌──────────────────┐       ┌──────────────────────────┐    │
│  │   Chat Tab        │       │  Data Management Tab      │    │
│  │  ui/chat.py       │       │  ui/management.py         │    │
│  └────────┬─────────┘       └────────────┬─────────────┘    │
└───────────┼──────────────────────────────┼──────────────────┘
            │                              │
            ▼                              ▼
┌───────────────────────┐   ┌──────────────────────────────┐
│   core/rag_engine.py  │   │  core/document_processor.py  │
│  ┌─────────────────┐  │   │  ┌──────────────────────┐    │
│  │  LangGraph App  │  │   │  │ PyPDF / Docx2txt /   │    │
│  │  (MemorySaver)  │  │   │  │ TextLoader / CSV      │    │
│  └────────┬────────┘  │   │  └──────────┬───────────┘    │
│           │           │   │             │                  │
│  ┌────────▼────────┐  │   │  ┌──────────▼───────────┐    │
│  │  Groq ChatGroq  │  │   │  │ RecursiveCharacter    │    │
│  │  llama-3.1-8b   │  │   │  │ TextSplitter          │    │
│  └─────────────────┘  │   │  └──────────┬───────────┘    │
│                        │   │             │                  │
│  ┌─────────────────┐  │   │  ┌──────────▼───────────┐    │
│  │   ChromaDB      │◄─┼───┼──│ HuggingFace Embeddings│    │
│  │  (persist/docs) │  │   │  │ multilingual-e5-large  │    │
│  └────────┬────────┘  │   │  └──────────────────────┘    │
└───────────┼───────────┘   └──────────────────────────────┘
            │
  ┌─────────▼──────────────────────────────┐
  │            Query Pipeline               │
  │  1. Retrieve top (K×3) candidates       │
  │  2. Re-rank with CrossEncoder           │  ← core/reranker.py
  │  3. Build context from top-K chunks     │
  │  4. Invoke LangGraph LLM node           │
  │  5. Score answer (hallucination check)  │  ← core/hallucination_detector.py
  └─────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
academic-qa-assistant/
├── app.py                         # Streamlit entry point
├── config.py                      # Central configuration
├── requirements.txt               # Python dependencies
├── pyproject.toml                 # Project metadata & pytest config
├── .env                           # API keys (NOT committed to Git)
├── .gitignore
│
├── core/                          # Business logic (no Streamlit imports)
│   ├── rag_engine.py              # LangGraph app, resource loaders, run_query()
│   ├── document_processor.py      # Ingestion pipeline (load → split → embed → store)
│   ├── hallucination_detector.py  # Cosine-similarity answer quality scorer
│   └── reranker.py                # Cross-encoder re-ranker
│
├── ui/                            # Streamlit UI layer
│   ├── chat.py                    # Chat Assistant tab
│   ├── management.py              # Data Management & Settings tab
│   └── styles.py                  # Custom CSS injection
│
├── data/
│   └── uploads/                   # Saved uploaded documents (gitignored)
│
├── docs/
│   └── chroma/                    # ChromaDB vector store (gitignored)
│
└── tests/
    └── unit/                      # Pytest unit tests
```

---

## 🚀 Quick Start

### 1. Prerequisites

- **Python 3.11+**
- A free **[Groq API key](https://console.groq.com/)** (no credit card required)

### 2. Clone the Repository

```bash
git clone https://github.com/<your-username>/academic-qa-assistant.git
cd academic-qa-assistant
```

### 3. Create a Virtual Environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

> ⚠️ **Note:** `torch` and the sentence-transformer models (~2–3 GB total) will be downloaded on first run and cached locally.

### 5. Configure API Key

Create a `.env` file in the project root:

```env
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

Get your key free at [console.groq.com](https://console.groq.com).

### 6. Run the Application

```bash
python -m streamlit run app.py
```

Open your browser at **http://localhost:8501** 🎉

---

## 📖 Usage Guide

### Step 1 — Upload Documents
1. Go to the **Data Management & Settings** tab.
2. Drag & drop or browse for **PDF, DOCX, TXT, or CSV** files.
3. Click **Process & Ingest Documents**.
4. Wait for the embedding process to complete (progress shown in the ingestion log).

### Step 2 — Ask Questions
1. Switch to the **Chat Assistant** tab.
2. Type your question in natural language.
3. The system will:
   - Retrieve the most relevant chunks from your documents
   - Re-rank them with a cross-encoder for precision
   - Generate a grounded answer via Groq LLM
   - Display a **confidence badge** (HIGH / MEDIUM / LOW)
   - Show expandable **source cards** with page references

### Step 3 — Tune Parameters
In the **Data Management & Settings** tab under **Model Parameters**:

| Parameter | Default | Effect |
|---|---|---|
| Temperature | `0.0` | Higher = more creative, Lower = more factual |
| Max Output Tokens | `1024` | Cap on response length |
| Retrieval Context (Top-K) | `6` | Chunks fed to LLM (fetches K×3 before re-ranking) |

---

## 🧠 How It Works

### RAG Pipeline

```
User Query
    │
    ▼
ChromaDB similarity_search (k = top_k × 3)
    │  [bi-encoder cosine similarity — fast retrieval]
    ▼
CrossEncoder re-ranking
    │  [ms-marco-MiniLM-L-6-v2 — joint query-document scoring]
    │  [keeps top_k most relevant chunks]
    ▼
Context assembly (top_k chunks joined)
    │
    ▼
LangGraph → ChatGroq (llama-3.1-8b-instant)
    │
    ▼
HallucinationDetector.score(context, answer)
    │  [chunk-level max cosine similarity — all-MiniLM-L6-v2]
    ▼
QueryResult(answer, sources, hallucination_score)
```

### Hallucination Detection

| Confidence | Label | Meaning |
|---|---|---|
| ≥ 0.40 | ✅ **HIGH** | Answer is closely grounded in retrieved chunks |
| ≥ 0.25 | ⚠️ **MEDIUM** | Partially grounded — may include inferences |
| < 0.25 | 🚨 **LOW** | Low overlap — try increasing Top-K or re-ingesting documents |

### Models Used

| Role | Model | Size | Runs Where |
|---|---|---|---|
| LLM | `llama-3.1-8b-instant` | — | Groq Cloud API |
| Embeddings | `intfloat/multilingual-e5-large` | ~2.2 GB | Local |
| Re-Ranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | ~80 MB | Local |
| Hallucination Scorer | `all-MiniLM-L6-v2` | ~80 MB | Local |

---

## ⚙️ Configuration

All settings are centralised in [`config.py`](config.py):

```python
EMBEDDING_MODEL_NAME  = "intfloat/multilingual-e5-large"
LLM_MODEL_NAME        = "llama-3.1-8b-instant"
CHROMA_PERSIST_DIR    = "docs/chroma"
UPLOAD_DIRECTORY      = "data/uploads"
CHUNK_SIZE            = 800     # characters
CHUNK_OVERLAP         = 100     # characters
DEFAULT_TEMPERATURE   = 0.0
DEFAULT_MAX_TOKENS    = 1024
DEFAULT_TOP_K         = 6
```

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

---

## 📦 Core Dependencies

| Package | Version | Purpose |
|---|---|---|
| `streamlit` | ≥1.55.0 | Web UI framework |
| `langchain` | ≥1.2.15 | LLM orchestration |
| `langgraph` | ≥1.1.9 | Stateful conversation graph |
| `langchain-groq` | ≥1.1.2 | Groq LLM integration |
| `chromadb` | ≥1.5.8 | Local vector database |
| `sentence-transformers` | ≥5.4.1 | Embeddings & re-ranking |
| `transformers` | ≥5.6.1 | HuggingFace model loading |
| `torch` | ≥2.10.0 | Neural network inference |
| `pypdf` | ≥6.10.2 | PDF document loading |
| `docx2txt` | ≥0.9 | DOCX document loading |
| `python-dotenv` | ≥1.2.2 | Environment variable management |

---

## 🔐 Security & Privacy

- **API keys** are stored in `.env` — never committed to Git
- **Uploaded documents** are stored locally in `data/uploads/` (gitignored)
- **ChromaDB** persists locally in `docs/chroma/` (gitignored)
- All embedding, re-ranking, and hallucination models run **fully locally** — your document contents never leave your machine

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m 'Add your feature'`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- [LangChain](https://langchain.com/) & [LangGraph](https://langchain-ai.github.io/langgraph/) for the orchestration framework
- [Groq](https://groq.com/) for ultra-fast LLM inference
- [ChromaDB](https://www.trychroma.com/) for the local vector store
- [Sentence Transformers](https://www.sbert.net/) for embedding and re-ranking models
- [Streamlit](https://streamlit.io/) for the web UI framework

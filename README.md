# ⚖️ LexiAI: Indian Legal RAG Assistant

> Ask questions about Indian law and get **cited answers** grounded in your own legal documents. It automatically maps old **IPC** section numbers to the new **BNS** (Bharatiya Nyaya Sanhita).

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![LangChain](https://img.shields.io/badge/LangChain-RAG-green)
![ChromaDB](https://img.shields.io/badge/Vector%20DB-Chroma-orange)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)

## ✨ Features

- **Document ingestion:** loads PDFs and TXT files from `data/raw/`
- **Smart chunking:** `RecursiveCharacterTextSplitter` with automatic section-number metadata for citations
- **Local embeddings:** Ollama `qwen3-embedding:0.6b` (free, runs on your machine)
- **Vector store:** ChromaDB with cosine similarity and full CRUD helpers
- **7 retrieval modes:** `similarity`, `threshold`, `mmr`, `bm25`, `hybrid`, `multi_query`, `compression`
- **IPC → BNS query rewriting:** "IPC 420" is automatically expanded to "BNS Section 318"
- **Grounded answers:** GPT-4o-mini answers only from retrieved passages, cites `[1]`, `[2]`, and abstains when the answer isn't found
- **Citation validation:** flags citations that don't match any retrieved passage
- **Works without an API key:** falls back to showing the best matching passages
- **Streamlit web UI** and a **CLI**

## 🏗️ Architecture

```
data/raw  →  Loaders  →  Chunking  →  Embeddings  →  ChromaDB
                                                        │
User question → IPC→BNS rewrite → Retriever ←───────────┘
                                      │
                               GPT-4o-mini (cited answer + disclaimer)
```

| Stage | Tool |
|---|---|
| Loaders | `PyPDFLoader`, `DirectoryLoader`, `TextLoader` |
| Chunking | `RecursiveCharacterTextSplitter` (500 / 50) |
| Embeddings | Ollama `qwen3-embedding:0.6b` |
| Vector store | `langchain_chroma.Chroma` |
| Retrievers | Similarity, Threshold, MMR, BM25, Ensemble (0.6 / 0.4), MultiQuery, Compression |
| Chat model | OpenAI `gpt-4o-mini` |
| UI | Streamlit |

## 📁 Project Structure

```
lexiai-indian-legal-rag/
├── app/
│   └── streamlit_app.py     # Web UI
├── data/raw/                # Put your legal PDFs / TXT files here
├── lexiai/
│   ├── config.py            # Settings from .env
│   ├── loaders.py           # Step 1: load documents
│   ├── chunking.py          # Step 2: split + metadata
│   ├── embeddings.py        # Step 3: Ollama embeddings
│   ├── vector_store.py      # Step 4: Chroma + CRUD
│   ├── retrievers.py        # Step 5: 7 retrieval modes
│   ├── rag.py               # Step 6: retrieval + answer generation
│   ├── legal.py             # IPC → BNS mapping
│   ├── ingest.py            # CLI: build the index
│   └── ask.py               # CLI: ask a question
├── tests/test_pipeline.py   # Offline tests
├── .env.example
└── requirements.txt
```

## 🚀 Getting Started

### 1. Clone and set up the environment

```bash
git clone https://github.com/<your-username>/lexiai-indian-legal-rag.git
cd lexiai-indian-legal-rag

python -m venv .venv
# Windows:    .venv\Scripts\activate
# Mac/Linux:  source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Install Ollama and pull the embedding model

Download Ollama from [ollama.com](https://ollama.com), then:

```bash
ollama pull qwen3-embedding:0.6b
```

### 3. Configure environment variables

```bash
cp .env.example .env
```

Open `.env` and add your own `OPENAI_API_KEY`. Without a key, the app still returns the best passages but no generated answer.

### 4. Add your documents

Place legal PDFs or TXT files in `data/raw/`. A small demo file (`demo_bns.txt`) is included for testing.

## ▶️ Usage

```bash
# Build the vector index
python -m lexiai.ingest

# Ask a question from the command line
python -m lexiai.ask "Punishment for cheating under IPC 420?" --mode hybrid

# Launch the web app
streamlit run app/streamlit_app.py

# Run tests (no API key or Ollama needed)
pytest -q
```

### Example

```
$ python -m lexiai.ask "What is the punishment for cheating under IPC 420?"

Rewritten: What is the punishment for cheating under IPC 420? Bharatiya Nyaya Sanhita BNS Section 318

Simple cheating is punished with imprisonment up to three years, or a fine,
or both [1]. This section replaces IPC Section 420 [1].

[1] demo_bns.txt Section 318
```

## 🔍 Retrieval Modes

| Mode | Description |
|---|---|
| `similarity` | Plain top-k vector search |
| `threshold` | Vector search with a minimum similarity score |
| `mmr` | Maximal Marginal Relevance, giving diverse results |
| `bm25` | Keyword search (custom tokenizer so "318" matches "Section 318.") |
| `hybrid` | MMR + BM25 ensemble (weights 0.6 / 0.4), **default** |
| `multi_query` | LLM generates query variations (needs API key) |
| `compression` | Filters out low-relevance passages using embeddings |

## ⚙️ Configuration

All settings live in `.env`:

| Variable | Default | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | none | Required for generated answers |
| `CHAT_MODEL` | `gpt-4o-mini` | LLM used for answers |
| `EMBED_MODEL` | `qwen3-embedding:0.6b` | Ollama embedding model |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `500` / `50` | Chunking settings |
| `TOP_K` | `4` | Passages retrieved |

> If you change the embedding model or chunk settings, re-run `python -m lexiai.ingest`.

## 💡 Tips

- For legal text, try `CHUNK_SIZE=300` so each section stays in its own chunk.
- Very small chunks (e.g. 100) hurt retrieval quality, so keep them at 300–500.
- Use official texts from [India Code](https://indiacode.nic.in) for real use.

## ⚠️ Disclaimer

- LexiAI provides **legal information for research, not legal advice**. Always verify against the official text.
- The included demo file is **paraphrased test data, not official law**.
- The IPC → BNS table in `legal.py` is a **starter table**. Verify every entry against the official correspondence table before relying on it.
- Never commit your `.env` file or API keys.

## 🛣️ Roadmap

- [ ] Full official IPC → BNS mapping table
- [ ] Support for CrPC → BNSS and Evidence Act → BSA
- [ ] Evaluation set (retrieval hit-rate, citation accuracy)
- [ ] Local LLM option (Ollama chat model) for fully offline use
- [ ] Docker setup

## 📄 License

MIT. Add a `LICENSE` file before publishing.

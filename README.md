# 📚 Alice in Wonderland LLM

[![CI](https://github.com/fadyabirached/alice-llm/actions/workflows/ci.yml/badge.svg)](https://github.com/fadyabirached/alice-llm/actions/workflows/ci.yml)

This project is a **Retrieval-Augmented Generation (RAG) chatbot** built to answer questions about the book *Alice in Wonderland*. Its real, default design runs entirely locally using **Ollama** and the open-source **Llama 3** model, optimized to provide accurate, source-based answers while minimizing AI hallucinations — see [Tech Stack](#️-tech-stack) and [Live demo vs. local design](#-live-demo-vs-local-design) for the one deliberate exception.

You can interact with it four ways: from the **command line** in the notebook, through a **local Streamlit app**, in **Google Colab** with a public URL, or via the **live demo** below.

---

## 🌐 Live Demo

**[Try it here →](https://alice-llm.streamlit.app/)**

The live demo swaps the LLM backend from Ollama to Groq's hosted API, since free
hosting can't run a local Ollama server. **This is a deployment-only exception —
the project's real design is Ollama end-to-end**, unchanged everywhere else. See
[Live demo vs. local design](#-live-demo-vs-local-design) for exactly what that
means and why.

---

## 🚀 Key Features

- **High-Quality Embeddings:** Uses `mxbai-embed-large` for highly relevant search results.
- **Advanced Retrieval:** Implements **Maximal Marginal Relevance (MMR)** to fetch context chunks that are both relevant and diverse.
- **Grounded Generation:** The LLM never states a fact the retrieved text doesn't support. If a question's premise contradicts the book (e.g. gets who-did-what backwards), it corrects the premise using the context instead of refusing or hallucinating — it only refuses outright when the context truly has nothing relevant.
- **Interactive Web Demo:** A local Streamlit app (`app.py`), or the same demo from Colab via Ngrok/LocalTunnel.
- **Fully Local & Open-Source:** Runs on your machine with Ollama and open-source models — no API keys required.
- **Tested Core Logic:** Chunking, prompt construction, and retriever configuration live in a small, unit-tested Python module (`src/rag.py`), not just notebook cells.
- **Colab Ready:** Easy setup and GPU access via a Google Colab notebook, if you'd rather not install anything locally.

---

## ⚙️ How It Works: The RAG Pipeline

### 1️⃣ Data Ingestion (Building the Knowledge Base)
The text of *Alice in Wonderland* is loaded, split into smaller, overlapping chunks, and converted into numerical embeddings using `mxbai-embed-large`. These embeddings are stored in a **FAISS vector store**, acting as a searchable library of the book's content.

```
[alice_in_wonderland.txt] -> Chunking -> Embedding -> [FAISS Vector Store]
```

### 2️⃣ Retrieval and Generation (Answering a Question)
When a user asks a question:

1. The query is embedded.
2. MMR searches the FAISS vector store for the most relevant and diverse chunks.
3. These chunks are added to a strict system prompt.
4. **Llama 3** generates a final answer based *only* on the provided context.

```
User Query -> Retrieval (MMR) -> [Relevant Chunks] -> Prompt + LLM -> Final Answer
```

---

## 🛠️ Tech Stack

- **Language:** Python 3.11
- **LLM Orchestration:** LangChain (`langchain`, `langchain-community`, `langchain-ollama`)
- **LLM Server (default/local):** Ollama
- **Generation Model (default/local):** Llama 3 (8B)
- **Embedding Model (default/local):** mxbai-embed-large
- **LLM Server (live demo only):** Groq's hosted API (`langchain-groq`) — see below
- **Embedding Model (live demo only):** `sentence-transformers/all-MiniLM-L6-v2`, run locally on CPU (`langchain-huggingface`)
- **Vector Store:** FAISS (`faiss-cpu`)
- **Interface:** Streamlit
- **Testing / CI:** pytest, ruff, GitHub Actions
- **Environment:** Runs locally, in Google Colab, or deployed to Streamlit Community Cloud

---

## 🔀 Live demo vs. local design

The default, documented, real design of this project is **Ollama end-to-end** —
generation and embeddings both run locally, no API keys, no data leaves your
machine. That's what `LLM_BACKEND=ollama` (the default if unset) gives you,
locally or via the notebook.

Free hosting platforms can't run a local Ollama server, so the **live demo
only** sets `LLM_BACKEND=groq`, which swaps two things in `src/rag.py`
(`get_llm` / `get_embeddings`):

| | Local (default) | Live demo (`LLM_BACKEND=groq`) |
|---|---|---|
| Generation | Ollama, Llama 3 8B, on your machine | Groq's hosted API, Llama 3 8B (`llama-3.1-8b-instant`) |
| Embeddings | Ollama, `mxbai-embed-large`, on your machine | `sentence-transformers/all-MiniLM-L6-v2`, still local/CPU, no API key |
| Requires | Ollama installed + models pulled | `GROQ_API_KEY` (free, from [console.groq.com](https://console.groq.com)) |
| Data leaves your machine? | No | Only the prompt + retrieved chunks, to Groq, for generation |

Everything else — chunking, FAISS, MMR retrieval, the grounded prompt — is
identical in both modes; `backend` is a parameter on `build_qa_system()`, not
a fork of the codebase. Embeddings stay local even in the live demo because
Groq doesn't serve an embeddings API — swapping to Groq only replaces the
generation half, not the whole pipeline.

---
<img width="1520" height="867" alt="image" src="https://github.com/user-attachments/assets/bd797140-b1ef-40c5-9a9e-f605084524e0" />

## 📁 Project Structure

```
alice-llm/
├── AliceWonderlandLLM.ipynb   # Interactive walkthrough / Colab entry point
├── app.py                     # Streamlit demo (imports src/rag.py); reads
│                               #   LLM_BACKEND to pick Ollama vs. Groq
├── alice_in_wonderland.txt    # Source text (the knowledge base)
├── src/
│   └── rag.py                 # Reusable, unit-tested RAG pipeline logic,
│                               #   incl. get_llm/get_embeddings backend swap
├── tests/
│   ├── test_rag.py             # Tests for pipeline logic that doesn't need
│   │                            #   a running Ollama server
│   └── test_backend_selection.py  # Tests for get_llm/get_embeddings
│                               #   (Ollama vs. Groq), mocked -- no real
│                               #   server, network call, or model download
├── requirements.txt            # Runtime deps: Ollama path + live-demo path
├── requirements-dev.txt        # + pytest, ruff (for local dev / CI)
└── .github/workflows/ci.yml    # Lints and runs the test suite on push/PR
```

`src/rag.py` factors out the chunking, prompt template, and retriever
configuration that both the notebook and `app.py` use, so that logic is
defined once, documented, and covered by tests — rather than duplicated
across notebook cells and an `app.py` written inline via `%%writefile`.

---

## 📖 Setup and Usage

### Option A — Run locally with Ollama (recommended)

**Prerequisites**

- Python 3.10+
- [Ollama](https://ollama.com/download) installed and running

**Steps**

1. **Clone the repository**
   ```bash
   git clone https://github.com/fadyabirached/alice-llm.git
   cd alice-llm
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Pull the required Ollama models**
   ```bash
   ollama pull llama3
   ollama pull mxbai-embed-large
   ```
   Make sure the Ollama server is running (`ollama serve`, or the desktop app) before continuing.

4. **Run the Streamlit demo**
   ```bash
   streamlit run app.py
   ```
   This opens the chatbot in your browser at `http://localhost:8501`.

   Alternatively, open `AliceWonderlandLLM.ipynb` locally (e.g. with
   `jupyter lab`) and run the "Build the RAG Pipeline" and "Ask Questions"
   cells for a command-line-style Q&A loop.

### Option B — Run in Google Colab

1. Go to [Google Colab](https://colab.research.google.com/) and upload
   `AliceWonderlandLLM.ipynb`.
2. Upload `alice_in_wonderland.txt` via the file explorer panel.
3. Run the notebook cells in order:
   - **Cell 1:** Install libraries & Ollama.
   - **Cell 2:** Start and verify the Ollama server.
   - **Cell 3:** Download `llama3` and `mxbai-embed-large`.
   - **Cell 4:** Build the RAG pipeline.
   - **Cell 5 / 6:** Ask questions interactively, or run a single clean test query.
   - **Cell 7 (optional):** Launch the Streamlit demo with a public URL via ngrok.

### Option C — Deploy your own live demo (Streamlit Community Cloud)

This deploys `app.py` with `LLM_BACKEND=groq` (see
[Live demo vs. local design](#-live-demo-vs-local-design) above for what that
changes and why free hosting needs it).

1. Get a free Groq API key at [console.groq.com](https://console.groq.com) →
   API Keys.
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with
   GitHub, click **New app**.
3. Pick this repo, branch `main`, main file path `app.py`.
4. Under **Advanced settings → Secrets**, add:
   ```toml
   LLM_BACKEND = "groq"
   GROQ_API_KEY = "your-groq-api-key-here"
   ```
5. Click **Deploy**. First load takes a minute or two (downloading the
   `all-MiniLM-L6-v2` embedding model and building the FAISS index); after
   that it's cached for the life of the app instance.

No code changes needed — `requirements.txt` already includes the
`langchain-groq` / `langchain-huggingface` / `sentence-transformers`
dependencies this path needs, alongside the Ollama ones the local path uses.

---

## 🧪 Running the Tests

The test suite covers the pipeline logic that doesn't require a running
Ollama server or downloaded models (text chunking, prompt formatting,
retriever configuration) — it's fast and runs entirely offline:

```bash
pip install -r requirements-dev.txt
pytest -v
ruff check .
```

Both are run automatically on every push/PR via GitHub Actions
(`.github/workflows/ci.yml`). Anything that requires a live Ollama server
(embedding, vector search end-to-end, generation) is intentionally left out
of CI and is meant to be exercised locally.

---

## 🔧 Customization

- **Different Book:** Replace `alice_in_wonderland.txt` and update the `file_path` passed to `src.rag.build_qa_system()`.
- **Different Model:** Change `llm_model` / `embedding_model` in `src.rag.build_qa_system()` (or `ChatOllama(model=...)` in the notebook) to any Ollama model you have pulled.
- **Different Backend:** Pass `backend="groq"` to `build_qa_system()`, or set the `LLM_BACKEND` env var before running `app.py` — see [Live demo vs. local design](#-live-demo-vs-local-design).
- **Retriever Tuning:** Adjust `k` and `fetch_k` via `src.rag.RetrieverConfig`.
- **Prompt Customization:** Edit `RAG_PROMPT_TEMPLATE` in `src/rag.py` for a different tone, style, or instructions.

---

## ⚖️ License

This project is licensed under the **MIT License**. See the `LICENSE` file for details.

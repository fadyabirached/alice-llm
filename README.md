# 📚 Alice in Wonderland LLM

[![CI](https://github.com/fadyabirached/alice-llm/actions/workflows/ci.yml/badge.svg)](https://github.com/fadyabirached/alice-llm/actions/workflows/ci.yml)

This project is a **Retrieval-Augmented Generation (RAG) chatbot** built to answer questions about the book *Alice in Wonderland*. It runs entirely locally using **Ollama** and the open-source **Llama 3** model, optimized to provide accurate, source-based answers while minimizing AI hallucinations.

You can interact with it three ways: from the **command line** in the notebook, through a **local Streamlit app**, or in **Google Colab** with a public URL.

---

## 🚀 Key Features

- **High-Quality Embeddings:** Uses `mxbai-embed-large` for highly relevant search results.
- **Advanced Retrieval:** Implements **Maximal Marginal Relevance (MMR)** to fetch context chunks that are both relevant and diverse.
- **Grounded Generation:** The LLM answers **only based on the provided text**, reducing hallucinations.
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
- **LLM Server:** Ollama
- **Generation Model:** Llama 3 (8B)
- **Embedding Model:** mxbai-embed-large
- **Vector Store:** FAISS (`faiss-cpu`)
- **Interface:** Streamlit
- **Testing / CI:** pytest, ruff, GitHub Actions
- **Environment:** Runs locally, or in Google Colab for easy setup / free GPU access

---

<img width="1550" height="902" alt="Screenshot 2025-08-15 124720" src="https://github.com/user-attachments/assets/684f7223-39a8-4b74-9490-04d670416579" />

## 📁 Project Structure

```
alice-llm/
├── AliceWonderlandLLM.ipynb   # Interactive walkthrough / Colab entry point
├── app.py                     # Local Streamlit demo (imports src/rag.py)
├── alice_in_wonderland.txt    # Source text (the knowledge base)
├── src/
│   └── rag.py                 # Reusable, unit-tested RAG pipeline logic
├── tests/
│   └── test_rag.py            # Tests for the parts of the pipeline that
│                               #   don't require a running Ollama server
├── requirements.txt            # Runtime dependencies
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
- **Retriever Tuning:** Adjust `k` and `fetch_k` via `src.rag.RetrieverConfig`.
- **Prompt Customization:** Edit `RAG_PROMPT_TEMPLATE` in `src/rag.py` for a different tone, style, or instructions.

---

## ⚖️ License

This project is licensed under the **MIT License**. See the `LICENSE` file for details.

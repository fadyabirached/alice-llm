# 📚 Alice in Wonderland LLM

This project is a **Retrieval-Augmented Generation (RAG) chatbot** built to answer questions about the book *Alice in Wonderland*. It runs entirely locally using **Ollama** and the powerful **Llama 3** model, optimized to provide accurate, source-based answers while minimizing AI hallucinations.  

You can now also **interact with the bot live in a web browser** via a Streamlit interface.

---

## 🚀 Key Features

- **High-Quality Embeddings:** Uses `mxbai-embed-large` for highly relevant search results.
- **Advanced Retrieval:** Implements **Maximal Marginal Relevance (MMR)** to fetch context chunks that are both relevant and diverse.
- **Grounded Generation:** The LLM answers **only based on the provided text**, reducing hallucinations.
- **Interactive Web Demo:** Streamlit + Ngrok/LocalTunnel lets you ask questions live in your browser.
- **Fully Local & Open-Source:** Runs on your machine with Ollama and open-source models.
- **Colab Ready:** Easy setup and GPU access via a Google Colab notebook.

---

## ⚙️ How It Works: The RAG Pipeline

### 1️⃣ Data Ingestion (Building the Knowledge Base)
The text of *Alice in Wonderland* is loaded, split into smaller, overlapping chunks, and converted into numerical embeddings using `mxbai-embed-large`. These embeddings are stored in a **FAISS vector store**, acting as a searchable library of the book’s content.

[alice_in_wonderland.txt] -> Chunking -> Embedding -> [FAISS Vector Store]


### 2️⃣ Retrieval and Generation (Answering a Question)
When a user asks a question:

1. The query is embedded.
2. MMR searches the FAISS vector store for the most relevant and diverse chunks.
3. These chunks are added to a strict system prompt.
4. **Llama 3** generates a final answer based *only* on the provided context.

User Query -> Retrieval (MMR) -> [Relevant Chunks] -> Prompt + LLM -> Final Answer

<img width="1550" height="902" alt="Screenshot 2025-08-15 124720" src="https://github.com/user-attachments/assets/684f7223-39a8-4b74-9490-04d670416579" />
---

## 🛠️ Tech Stack

- **Language:** Python 3
- **LLM Orchestration:** LangChain
- **LLM Server:** Ollama
- **Generation Model:** Llama 3 (8B)
- **Embedding Model:** mxbai-embed-large
- **Vector Store:** FAISS (Facebook AI Similarity Search)
- **Interface:** Streamlit
- **Environment:** Google Colab (optional for easy setup)

---

## 📖 Setup and Usage

### Prerequisites

- Google Account (for Colab)
- `alice_in_wonderland.txt` file

### Step-by-Step Instructions

1. **Clone the Repository**
   Download the `.ipynb` notebook and the text file from this repository.

2. **Open in Google Colab**
   - Go to [Google Colab](https://colab.research.google.com/)
   - Upload the `.ipynb` notebook.

3. **Upload the Book**
   - Click the **folder icon** in Colab to open the file explorer.
   - Drag and drop `alice_in_wonderland.txt`.

4. **Run the Notebook Cells in Order**
   - **Cell 1:** Install libraries & Ollama.
   - **Cell 2:** Start and verify the Ollama server.
   - **Cell 3:** Download `llama3` and `mxbai-embed-large`.
   - **Cell 4:** Build the RAG pipeline.
   - **Cell 5:** Start the Streamlit web demo (you’ll get a public URL via Ngrok/LocalTunnel).

5. **Interact with the Chatbot**
   - Open the URL printed in the notebook.
   - Ask questions directly in the browser — answers appear live.

---

## 🔧 Customization

- **Different Book:** Replace `alice_in_wonderland.txt` and update `file_path` in the notebook.
- **Different Model:** Change `ChatOllama(model="llama3", ...)` to any Ollama model you have.
- **Retriever Tuning:** Adjust `k` and `fetch_k` in `vectorstore.as_retriever()` for more/fewer chunks.
- **Prompt Customization:** Modify the `ChatPromptTemplate` for tone, style, or instructions.

---

## ⚖️ License

This project is licensed under the **MIT License**. See the `LICENSE` file for details.

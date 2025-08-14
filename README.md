# 📚 Alice in Wonderland Chatbot

This project is a Retrieval-Augmented Generation (RAG) chatbot built to answer questions about the book "Alice in Wonderland". It runs entirely locally using Ollama and the powerful Llama 3 model, and it's specifically optimized to provide accurate, source-based answers while minimizing AI hallucinations.

This repository documents the process of building a robust RAG system, from identifying common pitfalls (like poor embeddings and retrieval) to implementing effective solutions. It is designed to be run in a Google Colab notebook for easy setup and GPU access.

## 🚀 Key Features

- **High-Quality Embeddings:** Utilizes `mxbai-embed-large`, a top-performing open-source model specifically for generating embeddings, leading to highly relevant search results.
- **Advanced Retrieval:** Implements **Maximal Marginal Relevance (MMR)** to fetch context chunks that are both relevant to the query and diverse, providing the LLM with a richer, less redundant context.
- **Grounded Generation:** Employs a strict system prompt that commands the LLM to answer **only** based on the provided text, significantly reducing the risk of hallucination.
- **Fully Local & Open-Source:** Runs entirely on a local Ollama server with open-source models (Llama 3 and mxbai-embed-large), ensuring privacy and full control.
- **Colab Ready:** The entire project is contained within a single Google Colab notebook (`.ipynb`) designed for a seamless, step-by-step setup and execution.

## ⚙️ How It Works: The RAG Pipeline

The project follows a standard but optimized RAG architecture in two main phases:

**1. Data Ingestion (Building the Knowledge Base)**
The text of "Alice in Wonderland" is loaded, split into smaller, overlapping chunks, and then converted into numerical representations (embeddings) using the `mxbai-embed-large` model. These embeddings are stored in a FAISS vector store, which acts as a searchable library of the book's content.
[alice_in_wonderland.txt] -> Chunking -> Embedding -> [FAISS Vector Store]

**2. Retrieval and Generation (Answering a Question)**
When a user asks a question, the query is also embedded. The MMR algorithm then searches the FAISS vector store to find the most relevant and diverse chunks of text. These chunks are inserted into a carefully crafted prompt along with the original question and sent to the Llama 3 model, which generates a final answer based *only* on the provided context.

User Query -> Retrieval (MMR) -> [Relevant Chunks] -> Prompt + LLM (Llama 3) -> Final Answer


## 🛠️ Tech Stack

- **Language:** Python 3
- **LLM Orchestration:** LangChain
- **LLM Server:** Ollama
- **Generation Model:** Llama 3 (8B)
- **Embedding Model:** mxbai-embed-large
- **Vector Store:** FAISS (Facebook AI Similarity Search)
- **Environment:** Google Colab

## 📖 Setup and Usage

This project is designed to be run in a Google Colab notebook.

### Prerequisites

- A Google Account (for Google Colab).
- The text of the book, saved as `alice_in_wonderland.txt`.

### Step-by-Step Instructions

1.  **Clone the Repository (or Download Files)**
    Download the `.ipynb` notebook file and the `alice_in_wonderland.txt` file from this repository.

2.  **Open in Google Colab**
    - Go to [Google Colab](https://colab.research.google.com/).
    - Click on `File > Upload notebook` and select the `.ipynb` file you downloaded.

3.  **Upload the Data File**
    - In your Colab notebook, click on the **folder icon** on the left sidebar to open the file explorer.
    - Drag and drop your `alice_in_wonderland.txt` file into this file explorer.

4.  **Run the Notebook Cells in Order**
    The notebook is divided into logical, numbered cells. Run them one by one.

    -   **Cell 1: Install Libraries & Ollama:** Installs all necessary Python packages and the Ollama software.
    -   **Cell 2: Start & Verify Ollama Server:** Starts the Ollama server in the background and runs crucial checks to ensure it is running correctly before proceeding.
    -   **Cell 3: Download AI Models:** Downloads the `llama3` and `mxbai-embed-large` models to the Ollama server. This step may take several minutes.
    -   **Cell 4: Build the RAG Pipeline:** This is the main setup cell. It loads the book, splits it into chunks, creates the embeddings, and builds the `retrieval_chain` object that will be used for answering questions.
    -   **Cell 5: Ask Questions:** This final cell contains an interactive loop where you can ask questions about the book. It also includes an alternative single-shot test for clean, non-interactive testing.

## 🔧 Customization

It's easy to adapt this project for your own needs:

-   **Use a Different Book:** Simply replace `alice_in_wonderland.txt` with your own `.txt` file and update the `file_path` variable in the notebook.
-   **Try a Different Model:** In the setup cell (Cell 4), you can change the model name in `ChatOllama(model="llama3", ...)` to any other model you have downloaded via Ollama (e.g., `mistral`, `gemma`).
-   **Tweak the Retriever:** You can adjust the number of chunks retrieved by changing the `k` and `fetch_k` parameters in `vectorstore.as_retriever()`.
-   **Modify the Prompt:** The system prompt inside the `ChatPromptTemplate` can be adjusted to change the chatbot's behavior, tone, or instructions.

## ⚖️ License

This project is licensed under the MIT License. See the `LICENSE` file for details.

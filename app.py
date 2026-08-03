"""Streamlit demo for the Alice in Wonderland RAG chatbot.

Run locally (after starting Ollama and pulling the required models — see
README.md):

    streamlit run app.py

This is the local, non-Colab counterpart to the notebook's Streamlit cell:
same pipeline, built on top of the reusable `src.rag` module instead of
being redefined inline.

Backend: controlled by the LLM_BACKEND env var, defaulting to "ollama" (the
project's real design). The live Streamlit Cloud demo sets LLM_BACKEND=groq
instead, since free hosting can't run a local Ollama server — see README
"Live demo" section for why, and src/rag.py's get_llm/get_embeddings for
what that swap actually changes.
"""
import os

import streamlit as st

from src.rag import build_qa_system

BOOK_PATH = "alice_in_wonderland.txt"
BACKEND = os.getenv("LLM_BACKEND", "ollama").lower()


@st.cache_resource
def get_qa_chain():
    """Build (and cache) the retrieval chain for the running session."""
    return build_qa_system(BOOK_PATH, backend=BACKEND)


st.title("🐇 Chat with Alice in Wonderland")

if BACKEND == "groq":
    st.caption(
        "☁️ Live demo backend: generation via Groq's hosted Llama 3, "
        "embeddings run locally on CPU. The default local setup (see "
        "README) uses Ollama end-to-end instead — no external API."
    )
else:
    st.caption("🖥️ Running locally via Ollama — no data leaves this machine.")

st.info(
    "Ask any question about 'Alice's Adventures in Wonderland' and get "
    "answers grounded in the book's text."
)

if not os.path.exists(BOOK_PATH):
    st.warning(f"'{BOOK_PATH}' was not found next to app.py. Please add it to begin.")
    st.stop()

try:
    with st.spinner("Getting Wonderland ready for your questions (first run only)..."):
        qa_chain = get_qa_chain()
except Exception as e:
    if BACKEND == "groq":
        st.error(f"Could not prepare the book: {e}\n\nIs GROQ_API_KEY set correctly?")
    else:
        st.error(
            f"Could not prepare the book: {e}\n\n"
            "Is the Ollama server running locally with `llama3` and "
            "`mxbai-embed-large` pulled? See README.md for setup."
        )
    st.stop()

user_query = st.text_input(
    "Ask a question:", placeholder="e.g., Why did Alice follow the White Rabbit?"
)

if user_query:
    with st.spinner("Searching for answers in Wonderland..."):
        response = qa_chain.invoke({"input": user_query})
        st.header("Answer:", divider="rainbow")
        st.write(response["answer"])

        with st.expander("Show Context Used"):
            st.write(
                "These are the exact snippets from the book used to "
                "generate the answer above."
            )
            st.json(
                [
                    {"content": doc.page_content, "metadata": doc.metadata}
                    for doc in response["context"]
                ]
            )

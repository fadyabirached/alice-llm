"""Streamlit demo for the Alice in Wonderland RAG chatbot.

Run locally (after starting Ollama and pulling the required models — see
README.md):

    streamlit run app.py

This is the local, non-Colab counterpart to the notebook's Streamlit cell:
same pipeline, but built on top of the reusable `src.rag` module instead of
being redefined inline.
"""
import os

import streamlit as st

from src.rag import build_qa_system

BOOK_PATH = "alice_in_wonderland.txt"


@st.cache_resource
def get_qa_chain():
    """Build (and cache) the retrieval chain for the running session."""
    return build_qa_system(BOOK_PATH)


st.title("🐇 Chat with Alice in Wonderland")
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

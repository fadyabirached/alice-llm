"""Streamlit chat demo for the Alice in Wonderland RAG pipeline.

    streamlit run app.py

Built on the reusable `src.rag` module rather than redefining the pipeline
inline. The LLM_BACKEND env var picks the backend, defaulting to "ollama";
the hosted demo sets it to "groq" because free hosting cannot run a local
Ollama server. See README "Live demo" for what that swap changes.
"""
import os

import streamlit as st

from src.rag import build_qa_system

BOOK_PATH = "alice_in_wonderland.txt"
BACKEND = os.getenv("LLM_BACKEND", "ollama").lower()

EXAMPLE_QUESTIONS = [
    "Who is the White Rabbit?",
    "What happens at the Mad Hatter's tea party?",
    "Why does Alice change size?",
    "How does Alice end up in Wonderland?",
]

SNIPPET_CHARS = 320

st.set_page_config(
    page_title="Chat with Alice in Wonderland",
    page_icon="🐇",
    layout="centered",
)


@st.cache_resource
def get_qa_chain():
    """Build (and cache) the retrieval chain for the running session."""
    return build_qa_system(BOOK_PATH, backend=BACKEND)


st.title("🐇 Chat with Alice in Wonderland")
st.caption(
    "Ask anything about the book. Every answer comes from its actual text, "
    "so it will not invent plot points."
)

if not os.path.exists(BOOK_PATH):
    st.warning(f"'{BOOK_PATH}' was not found next to app.py. Please add it to begin.")
    st.stop()

try:
    with st.spinner("Getting the book ready (first run only)..."):
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

if "messages" not in st.session_state:
    st.session_state.messages = []


def show_sources(snippets):
    with st.expander(f"Passages used ({len(snippets)})"):
        for snippet in snippets:
            st.markdown(f"> {snippet}")


# Resolved before anything renders so that a question already in flight
# takes the suggestions off screen in the same pass, rather than leaving
# them sitting above the first answer.
typed = st.chat_input("Ask about Alice in Wonderland...")
question = typed or st.session_state.pop("pending", None)

# Suggestions only help before the conversation starts. After that the chat
# box is the obvious thing to use.
if not st.session_state.messages and not question:
    st.write("**Not sure where to start?**")
    columns = st.columns(2)
    for index, suggestion in enumerate(EXAMPLE_QUESTIONS):
        if columns[index % 2].button(suggestion, use_container_width=True):
            st.session_state.pending = suggestion
            st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message.get("sources"):
            show_sources(message["sources"])

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Looking through the book..."):
            response = qa_chain.invoke({"input": question})

        answer = response["answer"]
        sources = [
            " ".join(doc.page_content.split())[:SNIPPET_CHARS] + "..."
            for doc in response["context"]
        ]

        st.write(answer)
        show_sources(sources)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )

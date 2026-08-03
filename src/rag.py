"""Reusable RAG (Retrieval-Augmented Generation) building blocks.

This module factors out the core pipeline logic used by the
``AliceWonderlandLLM.ipynb`` notebook and the Streamlit demo (``app.py``)
into a plain, importable, testable library.

Functions are split into two groups:

* **Pure / local** — text chunking, prompt construction, retriever
  configuration, and result formatting. These do not touch the network
  or a local model and are covered by the test suite in ``tests/``.
* **Ollama-backed** — building embeddings, a FAISS vector store, and the
  full retrieval chain. These require a running Ollama server with the
  ``mxbai-embed-large`` and ``llama3`` models pulled locally, so they are
  not exercised in CI and their heavier imports are deferred until the
  functions are actually called.

Example
-------
>>> from src.rag import build_qa_system
>>> chain = build_qa_system("alice_in_wonderland.txt")  # doctest: +SKIP
>>> chain.invoke({"input": "Who is the White Rabbit?"})  # doctest: +SKIP
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- Defaults, kept in one place so the notebook, the Streamlit app, and ---
# --- the tests all agree on the pipeline's configuration.               ---

DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200

DEFAULT_EMBEDDING_MODEL = "mxbai-embed-large"
DEFAULT_LLM_MODEL = "llama3"

DEFAULT_RETRIEVER_SEARCH_TYPE = "mmr"
DEFAULT_RETRIEVER_K = 8
DEFAULT_RETRIEVER_FETCH_K = 20

# The strict, grounded-answer prompt used across the notebook and the app.
# Built via concatenation (rather than one long line) purely to satisfy the
# line-length linter -- the rendered text is identical to a single sentence.
RAG_PROMPT_TEMPLATE = (
    "Use the following pieces of context to answer the question at the end.\n"
    "You must answer based ONLY on the provided context.\n"
    "If the answer is not contained within the text provided, you must say "
    '"I cannot find that information in the provided text."\n'
    "Do not provide any information or commentary outside of the given context.\n"
    "\n"
    "<context>\n"
    "{context}\n"
    "</context>\n"
    "\n"
    "Question: {input}\n"
)


# --------------------------------------------------------------------------
# Pure / local helpers (no network, no local model required)
# --------------------------------------------------------------------------


def split_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[str]:
    """Split a raw text string into overlapping chunks.

    Pure-Python helper (no I/O, no network) so it can be unit tested
    directly, independent of the document loader.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_text(text)


def load_and_split_documents(
    file_path: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[Document]:
    """Load a text file and split it into LangChain ``Document`` chunks.

    Raises ``FileNotFoundError`` if ``file_path`` does not exist, mirroring
    the guard the notebook and app perform before building the index.
    """
    from langchain_community.document_loaders import TextLoader

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"'{file_path}' not found.")

    loader = TextLoader(file_path, encoding="utf-8")
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(documents)


def build_prompt_template() -> ChatPromptTemplate:
    """Build the strict, grounded-answer prompt used by the RAG chain."""
    return ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)


@dataclass(frozen=True)
class RetrieverConfig:
    """Configuration for the FAISS retriever."""

    search_type: str = DEFAULT_RETRIEVER_SEARCH_TYPE
    k: int = DEFAULT_RETRIEVER_K
    fetch_k: int = DEFAULT_RETRIEVER_FETCH_K

    def as_search_kwargs(self) -> dict:
        return {"k": self.k, "fetch_k": self.fetch_k}


def build_retriever(vectorstore, config: Optional[RetrieverConfig] = None):
    """Build an MMR retriever from a FAISS vectorstore.

    ``vectorstore`` is expected to be a ``FAISS`` instance (or anything with
    a compatible ``as_retriever`` method), so this only touches the network
    if the retriever is actually invoked.
    """
    config = config or RetrieverConfig()
    return vectorstore.as_retriever(
        search_type=config.search_type, search_kwargs=config.as_search_kwargs()
    )


def format_retrieved_chunks(docs, preview_chars: int = 120) -> List[str]:
    """Format retrieved documents into short debug/CLI preview strings."""
    return [
        f"[CHUNK {i + 1}]: {doc.page_content[:preview_chars]}..."
        for i, doc in enumerate(docs)
    ]


# --------------------------------------------------------------------------
# Ollama-backed helpers (require a running local Ollama server)
# --------------------------------------------------------------------------


def build_vectorstore(splits: List[Document], embeddings):
    """Embed ``splits`` and build a FAISS vector store.

    ``embeddings`` must be an embeddings backend (e.g. ``OllamaEmbeddings``)
    reachable at call time.
    """
    from langchain_community.vectorstores import FAISS

    return FAISS.from_documents(documents=splits, embedding=embeddings)


def build_rag_chain(
    vectorstore,
    llm=None,
    llm_model: str = DEFAULT_LLM_MODEL,
    retriever_config: Optional[RetrieverConfig] = None,
):
    """Assemble the full retrieval chain from a vectorstore and an LLM.

    Pass an existing ``llm`` (e.g. for testing with a fake model) or let it
    default to ``ChatOllama(model=llm_model, temperature=0)``.
    """
    from langchain.chains import create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain

    if llm is None:
        from langchain_ollama import ChatOllama

        llm = ChatOllama(model=llm_model, temperature=0)

    retriever = build_retriever(vectorstore, retriever_config)
    prompt = build_prompt_template()
    document_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, document_chain)


def build_qa_system(
    file_path: str = "alice_in_wonderland.txt",
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    llm_model: str = DEFAULT_LLM_MODEL,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    retriever_config: Optional[RetrieverConfig] = None,
):
    """End-to-end pipeline: load, split, embed, index, and wire up the chain.

    Requires a running Ollama server with ``embedding_model`` and
    ``llm_model`` pulled locally (see the README for setup). This is the
    function both the notebook and ``app.py`` call to get a ready-to-use
    ``retrieval_chain``.
    """
    from langchain_ollama import OllamaEmbeddings

    splits = load_and_split_documents(file_path, chunk_size, chunk_overlap)
    embeddings = OllamaEmbeddings(model=embedding_model)
    vectorstore = build_vectorstore(splits, embeddings)
    return build_rag_chain(
        vectorstore, llm_model=llm_model, retriever_config=retriever_config
    )

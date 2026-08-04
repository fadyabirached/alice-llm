"""Reusable RAG (Retrieval-Augmented Generation) building blocks.

This module factors out the core pipeline logic used by the
``AliceWonderlandLLM.ipynb`` notebook and the Streamlit demo (``app.py``)
into a plain, importable, testable library.

Functions are split into two groups:

* **Pure / local**: text chunking, prompt construction, retriever
  configuration, and result formatting. These do not touch the network
  or a local model and are covered by the test suite in ``tests/``.
* **Ollama-backed**: building embeddings, a FAISS vector store, and the
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

# Used only when backend="groq" -- the live Streamlit Cloud demo, where free
# hosting can't run a local Ollama server. This is NOT the project's real
# design (see README): it's a swap for that one deployed instance only.
# Groq occasionally retires/renames models; override with GROQ_MODEL if this
# one stops working -- see https://console.groq.com/docs/models.
DEFAULT_DEPLOY_LLM_MODEL = "llama-3.1-8b-instant"
# Groq doesn't serve embeddings, so the "groq" backend pairs Groq generation
# with a small CPU-friendly sentence-transformers model instead (downloaded
# once from Hugging Face, no API key needed for this part).
DEFAULT_DEPLOY_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

DEFAULT_RETRIEVER_SEARCH_TYPE = "mmr"
DEFAULT_RETRIEVER_K = 8
DEFAULT_RETRIEVER_FETCH_K = 20

# The strict, grounded-answer prompt used across the notebook and the app.
# Built via concatenation (rather than one long line) purely to satisfy the
# line-length linter -- the rendered text is identical to a single sentence.
#
# Grounded means "never state a fact the context doesn't support" -- it does
# NOT mean "refuse whenever the question's own premise is wrong." If the
# context contradicts something the question assumed (e.g. asks who followed
# whom, backwards), the correct grounded answer is to correct it using the
# context, not to flatly refuse when the real answer was sitting right there.
# The refusal line is reserved for when the context truly has nothing
# relevant -- not as a catch-all for "the question phrased it wrong."
RAG_PROMPT_TEMPLATE = (
    "Use the following pieces of context to answer the question at the end.\n"
    "You must answer based ONLY on the provided context -- never state a "
    "fact the context doesn't support.\n"
    "If the context contradicts a premise in the question (for example, it "
    "gets a fact backwards), say so explicitly and give the correct answer "
    "using the context, instead of just refusing.\n"
    "If the context truly contains nothing relevant to the question, you "
    "must say "
    '"I cannot find that information in the provided text." Do not guess.\n'
    "Do not provide any information or commentary outside of the given "
    "context.\n"
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
# Backend-selected helpers (require either a local Ollama server, or --
# for backend="groq", used only by the live demo -- a GROQ_API_KEY)
# --------------------------------------------------------------------------


def get_llm(backend: str = "ollama", model: Optional[str] = None, temperature: float = 0):
    """Build the chat LLM for the given backend.

    * ``"ollama"`` (default): ``ChatOllama`` against a local Ollama server --
      the project's real, documented design (see README).
    * ``"groq"``: ``ChatGroq`` calling Groq's hosted API. Used only for the
      live Streamlit Cloud demo, where free hosting can't run a local
      Ollama server. Requires ``GROQ_API_KEY`` in the environment.
    """
    backend = backend.lower()
    if backend == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(model=model or DEFAULT_LLM_MODEL, temperature=temperature)

    if backend == "groq":
        if not os.getenv("GROQ_API_KEY"):
            raise RuntimeError(
                "backend='groq' requires a GROQ_API_KEY environment variable "
                "(free key at https://console.groq.com)."
            )
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=model or os.getenv("GROQ_MODEL", DEFAULT_DEPLOY_LLM_MODEL),
            temperature=temperature,
        )

    raise ValueError(f"Unknown LLM backend {backend!r}; use 'ollama' or 'groq'.")


def get_embeddings(backend: str = "ollama", model: Optional[str] = None):
    """Build the embeddings backend for the given backend name.

    * ``"ollama"`` (default): ``OllamaEmbeddings`` against a local Ollama
      server, same as ``get_llm``.
    * ``"groq"``: Groq doesn't serve an embeddings API, so this pairs Groq
      generation with a small local CPU sentence-transformers model instead
      -- no API key needed for this half, just a one-time download from
      Hugging Face on first run.
    """
    backend = backend.lower()
    if backend == "ollama":
        from langchain_ollama import OllamaEmbeddings

        return OllamaEmbeddings(model=model or DEFAULT_EMBEDDING_MODEL)

    if backend == "groq":
        from langchain_huggingface import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(model_name=model or DEFAULT_DEPLOY_EMBEDDING_MODEL)

    raise ValueError(f"Unknown embeddings backend {backend!r}; use 'ollama' or 'groq'.")


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
    llm_model: Optional[str] = None,
    backend: str = "ollama",
    retriever_config: Optional[RetrieverConfig] = None,
):
    """Assemble the full retrieval chain from a vectorstore and an LLM.

    Pass an existing ``llm`` (e.g. for testing with a fake model) or let it
    be built from ``backend``/``llm_model`` via :func:`get_llm`.
    """
    from langchain.chains import create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain

    if llm is None:
        llm = get_llm(backend=backend, model=llm_model)

    retriever = build_retriever(vectorstore, retriever_config)
    prompt = build_prompt_template()
    document_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, document_chain)


def build_qa_system(
    file_path: str = "alice_in_wonderland.txt",
    embedding_model: Optional[str] = None,
    llm_model: Optional[str] = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    backend: str = "ollama",
    retriever_config: Optional[RetrieverConfig] = None,
):
    """End-to-end pipeline: load, split, embed, index, and wire up the chain.

    ``backend="ollama"`` (default) requires a running Ollama server with
    ``embedding_model``/``llm_model`` pulled locally -- the project's real
    design (see README). ``backend="groq"`` is used only by the live demo
    deployment; see :func:`get_llm` / :func:`get_embeddings`. This is the
    function the notebook, ``app.py``, and the live demo all call to get a
    ready-to-use ``retrieval_chain``.
    """
    splits = load_and_split_documents(file_path, chunk_size, chunk_overlap)
    embeddings = get_embeddings(backend=backend, model=embedding_model)
    vectorstore = build_vectorstore(splits, embeddings)
    return build_rag_chain(
        vectorstore, llm_model=llm_model, backend=backend, retriever_config=retriever_config
    )

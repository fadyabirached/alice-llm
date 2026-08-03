"""Tests for src.rag.

These only cover logic that runs entirely locally: text chunking, prompt
construction, retriever configuration, and result formatting. Nothing here
requires a running Ollama server, a downloaded model, or network access.
"""
import pytest
from langchain_core.documents import Document

from src.rag import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    RAG_PROMPT_TEMPLATE,
    RetrieverConfig,
    build_prompt_template,
    build_retriever,
    format_retrieved_chunks,
    load_and_split_documents,
    split_text,
)

# --------------------------------------------------------------------------
# split_text
# --------------------------------------------------------------------------


def test_split_text_returns_single_chunk_for_short_text():
    text = "Alice was beginning to get very tired."
    chunks = split_text(text, chunk_size=1000, chunk_overlap=200)
    assert chunks == [text]


def test_split_text_splits_long_text_into_multiple_overlapping_chunks():
    # Build text long enough to force more than one chunk at the default size.
    paragraph = "Alice fell down the rabbit hole. " * 100
    chunks = split_text(
        paragraph, chunk_size=DEFAULT_CHUNK_SIZE, chunk_overlap=DEFAULT_CHUNK_OVERLAP
    )

    assert len(chunks) > 1
    # Every chunk should respect the configured chunk size.
    assert all(len(chunk) <= DEFAULT_CHUNK_SIZE for chunk in chunks)
    # Consecutive chunks should overlap (share trailing/leading text), which
    # is what gives the retriever context continuity across chunk boundaries.
    assert chunks[0][-50:] in paragraph
    assert chunks[1][:50] in paragraph


def test_split_text_respects_custom_chunk_size():
    text = "word " * 500  # 2500 characters
    small_chunks = split_text(text, chunk_size=100, chunk_overlap=20)
    large_chunks = split_text(text, chunk_size=1000, chunk_overlap=20)

    assert len(small_chunks) > len(large_chunks)
    assert all(len(chunk) <= 100 for chunk in small_chunks)


# --------------------------------------------------------------------------
# load_and_split_documents
# --------------------------------------------------------------------------


def test_load_and_split_documents_raises_for_missing_file(tmp_path):
    missing_path = tmp_path / "does_not_exist.txt"
    with pytest.raises(FileNotFoundError):
        load_and_split_documents(str(missing_path))


def test_load_and_split_documents_splits_a_real_file(tmp_path):
    content = "Down the rabbit hole. " * 200
    book = tmp_path / "book.txt"
    book.write_text(content, encoding="utf-8")

    splits = load_and_split_documents(str(book), chunk_size=200, chunk_overlap=20)

    assert len(splits) > 1
    assert all(isinstance(doc, Document) for doc in splits)
    assert all(len(doc.page_content) <= 200 for doc in splits)


# --------------------------------------------------------------------------
# build_prompt_template
# --------------------------------------------------------------------------


def test_build_prompt_template_has_context_and_input_variables():
    prompt = build_prompt_template()
    assert set(prompt.input_variables) == {"context", "input"}


def test_prompt_template_formats_with_context_and_question():
    prompt = build_prompt_template()
    messages = prompt.format_messages(
        context="Alice met the Cheshire Cat.", input="Who did Alice meet?"
    )

    rendered = messages[0].content
    assert "Alice met the Cheshire Cat." in rendered
    assert "Who did Alice meet?" in rendered
    # The grounding instruction must survive verbatim -- it's what keeps the
    # LLM from hallucinating answers outside the retrieved context.
    assert "I cannot find that information in the provided text." in rendered


def test_rag_prompt_template_constant_defines_expected_placeholders():
    assert "{context}" in RAG_PROMPT_TEMPLATE
    assert "{input}" in RAG_PROMPT_TEMPLATE


def test_prompt_instructs_correcting_false_premises_instead_of_refusing():
    """Grounded means "never state a fact the context doesn't support" --
    not "refuse whenever the question's phrasing is wrong." If a question
    gets a fact backwards but the context contains the real answer, the
    prompt must tell the model to correct it using the context rather than
    fall back to the generic refusal line."""
    lowered = RAG_PROMPT_TEMPLATE.lower()
    assert "contradicts" in lowered
    assert "correct" in lowered
    # The refusal line must still exist for genuinely irrelevant context --
    # this isn't removing the anti-hallucination guarantee, just narrowing
    # when the refusal applies.
    assert "i cannot find that information in the provided text." in lowered


# --------------------------------------------------------------------------
# RetrieverConfig / build_retriever
# --------------------------------------------------------------------------


def test_retriever_config_defaults():
    config = RetrieverConfig()
    assert config.search_type == "mmr"
    assert config.as_search_kwargs() == {"k": 8, "fetch_k": 20}


def test_retriever_config_custom_values():
    config = RetrieverConfig(search_type="mmr", k=4, fetch_k=10)
    assert config.as_search_kwargs() == {"k": 4, "fetch_k": 10}


class _FakeVectorStore:
    """Minimal stand-in for a FAISS vectorstore, to test wiring without FAISS."""

    def __init__(self):
        self.last_call = None

    def as_retriever(self, search_type, search_kwargs):
        self.last_call = (search_type, search_kwargs)
        return f"retriever(search_type={search_type}, search_kwargs={search_kwargs})"


def test_build_retriever_passes_config_through_to_vectorstore():
    store = _FakeVectorStore()
    build_retriever(store, RetrieverConfig(k=3, fetch_k=9))

    assert store.last_call == ("mmr", {"k": 3, "fetch_k": 9})


def test_build_retriever_uses_default_config_when_none_given():
    store = _FakeVectorStore()
    build_retriever(store)

    assert store.last_call == ("mmr", {"k": 8, "fetch_k": 20})


# --------------------------------------------------------------------------
# format_retrieved_chunks
# --------------------------------------------------------------------------


def test_format_retrieved_chunks_previews_each_document():
    docs = [
        Document(page_content="A" * 200),
        Document(page_content="short"),
    ]
    formatted = format_retrieved_chunks(docs, preview_chars=10)

    assert len(formatted) == 2
    assert formatted[0] == f"[CHUNK 1]: {'A' * 10}..."
    assert formatted[1] == "[CHUNK 2]: short..."


def test_format_retrieved_chunks_handles_empty_list():
    assert format_retrieved_chunks([]) == []

"""Tests for the backend-selection helpers in src.rag: get_llm and
get_embeddings, which switch between the project's default local Ollama
setup and the Groq-backed live demo. These monkeypatch the actual
ChatOllama/ChatGroq/OllamaEmbeddings/HuggingFaceEmbeddings constructors so
no real Ollama server, network call, or model download happens.
"""
import pytest

from src.rag import (
    DEFAULT_DEPLOY_EMBEDDING_MODEL,
    DEFAULT_DEPLOY_LLM_MODEL,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_LLM_MODEL,
    get_embeddings,
    get_llm,
)


class _FakeChatModel:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _FakeEmbeddings:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


# --------------------------------------------------------------------------
# get_llm
# --------------------------------------------------------------------------


def test_get_llm_ollama_backend_uses_default_model(monkeypatch):
    import langchain_ollama

    monkeypatch.setattr(langchain_ollama, "ChatOllama", _FakeChatModel)
    llm = get_llm(backend="ollama")

    assert isinstance(llm, _FakeChatModel)
    assert llm.kwargs["model"] == DEFAULT_LLM_MODEL
    assert llm.kwargs["temperature"] == 0


def test_get_llm_ollama_backend_respects_custom_model(monkeypatch):
    import langchain_ollama

    monkeypatch.setattr(langchain_ollama, "ChatOllama", _FakeChatModel)
    llm = get_llm(backend="ollama", model="llama3.1")

    assert llm.kwargs["model"] == "llama3.1"


def test_get_llm_groq_backend_requires_api_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        get_llm(backend="groq")


def test_get_llm_groq_backend_uses_default_deploy_model(monkeypatch):
    import langchain_groq

    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.delenv("GROQ_MODEL", raising=False)
    monkeypatch.setattr(langchain_groq, "ChatGroq", _FakeChatModel)

    llm = get_llm(backend="groq")

    assert isinstance(llm, _FakeChatModel)
    assert llm.kwargs["model"] == DEFAULT_DEPLOY_LLM_MODEL


def test_get_llm_groq_backend_respects_groq_model_env_override(monkeypatch):
    import langchain_groq

    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    monkeypatch.setattr(langchain_groq, "ChatGroq", _FakeChatModel)

    llm = get_llm(backend="groq")

    assert llm.kwargs["model"] == "llama-3.3-70b-versatile"


def test_get_llm_backend_is_case_insensitive(monkeypatch):
    import langchain_ollama

    monkeypatch.setattr(langchain_ollama, "ChatOllama", _FakeChatModel)
    llm = get_llm(backend="OLLAMA")

    assert isinstance(llm, _FakeChatModel)


def test_get_llm_unknown_backend_raises():
    with pytest.raises(ValueError, match="Unknown LLM backend"):
        get_llm(backend="openai")


# --------------------------------------------------------------------------
# get_embeddings
# --------------------------------------------------------------------------


def test_get_embeddings_ollama_backend_uses_default_model(monkeypatch):
    import langchain_ollama

    monkeypatch.setattr(langchain_ollama, "OllamaEmbeddings", _FakeEmbeddings)
    embeddings = get_embeddings(backend="ollama")

    assert isinstance(embeddings, _FakeEmbeddings)
    assert embeddings.kwargs["model"] == DEFAULT_EMBEDDING_MODEL


def test_get_embeddings_groq_backend_uses_local_sentence_transformers_model(monkeypatch):
    import langchain_huggingface

    monkeypatch.setattr(langchain_huggingface, "HuggingFaceEmbeddings", _FakeEmbeddings)
    embeddings = get_embeddings(backend="groq")

    assert isinstance(embeddings, _FakeEmbeddings)
    assert embeddings.kwargs["model_name"] == DEFAULT_DEPLOY_EMBEDDING_MODEL


def test_get_embeddings_unknown_backend_raises():
    with pytest.raises(ValueError, match="Unknown embeddings backend"):
        get_embeddings(backend="openai")

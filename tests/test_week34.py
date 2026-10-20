"""Persistence, provider consistency, upserts, metadata and retrieval boundaries."""

from pathlib import Path

import pytest

from rag.config import settings
from rag.ingestion.models import Chunk
from rag.providers.gemini import GeminiEmbeddings
from rag.retrieval.retriever import Retriever
from rag.vectorstore.chroma_store import ChromaStore


def test_persist_reopen_and_idempotent_upsert(tmp_path, mock_embeddings):
    path = tmp_path / ".chroma"
    store = ChromaStore(mock_embeddings, path)
    chunk = Chunk("original", "doc.md", 3, {"nested": {"a": 1}, "source": "wrong"})
    store.add([chunk])
    chunk.text = "updated"
    store.add([chunk])
    reopened = ChromaStore(mock_embeddings, path)
    assert reopened.count() == 1
    result = reopened.query("updated", k=5)[0]
    assert result.text == "updated" and result.id == chunk.id
    assert result.metadata["nested"] == "{'a': 1}"
    assert mock_embeddings.calls[-1] == ["updated"]
    reopened.clear()
    reopened.add([chunk])
    assert reopened.query("updated")[0].id == chunk.id


def test_empty_store_and_invalid_k_do_not_embed(chroma_store, mock_embeddings):
    chroma_store.add([])
    assert chroma_store.query("question") == []
    assert chroma_store.query("question", k=0) == []
    assert chroma_store.query("question", k=-1) == []
    assert chroma_store.query("   ") == []
    assert mock_embeddings.calls == []


def test_wrong_embedding_count_fails_before_writing(chroma_store, mock_embeddings, monkeypatch):
    monkeypatch.setattr(mock_embeddings, "embed", lambda texts: [[1.0] * 16])
    with pytest.raises(ValueError, match="one vector per chunk"):
        chroma_store.add([Chunk("a", "s", 0), Chunk("b", "s", 1)])
    assert chroma_store.count() == 0


def test_retriever_default_k_and_source_filter(populated_store, sample_document):
    results = Retriever(populated_store).search("RAG", filters={"source": sample_document.source})
    assert 0 < len(results) <= settings.top_k
    assert all(c.source == sample_document.source for c in results)
    assert Retriever(populated_store).search("RAG", filters={"source": "missing"}) == []


def test_gemini_embeds_each_chunk_separately(monkeypatch):
    calls = []
    provider = GeminiEmbeddings()

    def embed_one(text):
        calls.append(text)
        return [1.0, 2.0]

    monkeypatch.setattr(provider, "_embed_one", embed_one)
    assert provider.embed(["first", "second"]) == [[1.0, 2.0], [1.0, 2.0]]
    assert calls == ["first", "second"]


def test_workshop_corpus_is_47_vectors(tmp_path, mock_embeddings):
    from rag.ingestion.chunker import Chunker
    from rag.ingestion.loader import DocumentLoader

    documents = DocumentLoader().load(Path("data/workshop_corpus"))
    chunks = Chunker(size=500, overlap=50).chunk(documents)
    assert len(documents) == 7 and len(chunks) == 47
    store = ChromaStore(mock_embeddings, tmp_path / ".chroma")
    store.add(chunks)
    store.add(chunks)
    assert store.count() == 47
    assert len(store.query(chunks[0].text, k=5)) == 5

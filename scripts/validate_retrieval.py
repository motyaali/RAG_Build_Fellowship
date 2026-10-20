"""Validate Workshop 3/4 with free local ONNX embeddings; no API keys or LLM calls.

First run downloads the public all-MiniLM-L6-v2 model (~80 MB).
Uses a separate .chroma-local directory so it cannot mix with Gemini vectors.
"""

from __future__ import annotations

import json
from pathlib import Path

from rag.ingestion.chunker import Chunker
from rag.ingestion.loader import DocumentLoader

# Backward-compatible import for the executed Week 4 notebook.
from rag.providers.local_onnx import LocalONNXEmbeddings as LocalValidationEmbeddings
from rag.retrieval.retriever import Retriever
from rag.vectorstore.chroma_store import ChromaStore


def main() -> None:
    documents = DocumentLoader().load(Path("data/workshop_corpus"))
    chunks = Chunker(size=500, overlap=50).chunk(documents)
    assert len(documents) == 7 and len(chunks) == 47
    embeddings = LocalValidationEmbeddings()
    store = ChromaStore(embeddings, persist_dir=Path(".chroma-local"))
    store.add(chunks)
    assert store.count() == 47
    store.add(chunks)
    assert store.count() == 47, "upsert must not duplicate chunks"
    reopened = ChromaStore(embeddings, persist_dir=Path(".chroma-local"))
    assert reopened.count() == 47
    retriever = Retriever(reopened)
    report = {
        "model": "all-MiniLM-L6-v2 (local ONNX)",
        "dimension": 384,
        "documents": len(documents),
        "vectors": reopened.count(),
        "queries": [],
    }
    for question, expected in [
        ("What is retrieval augmented generation?", "rag_primer.md"),
        ("How does ChromaDB persist vectors?", "chromadb_overview.txt"),
        ("Why do similar texts have nearby embeddings?", "embeddings_explained.md"),
        ("What is the Apache License?", "apache_foundation_faq.md"),
    ]:
        for k in (3, 5):
            results = retriever.search(question, k=k)
            assert len(results) == k
            assert any(Path(c.source).name == expected for c in results), question
            report["queries"].append(
                {
                    "question": question,
                    "k": k,
                    "results": [
                        {"source": c.source, "chunk_index": c.chunk_index, "text": c.text}
                        for c in results
                    ],
                }
            )
    source = chunks[0].source
    filtered = retriever.search("Apache", filters={"source": source})
    assert filtered and all(c.source == source for c in filtered)
    assert retriever.search("   ") == []
    report["metadata_filter_passed"] = True
    Path("evidence").mkdir(exist_ok=True)
    Path("evidence/local-retrieval.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        "PASS: 7 documents, 47 persistent vectors, idempotent upsert, 8 top-k queries, metadata filter"
    )


if __name__ == "__main__":
    main()

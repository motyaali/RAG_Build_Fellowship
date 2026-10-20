"""Tests for the loader and chunker.

Enabled for the completed Workshop 2-4 modules.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rag.ingestion.chunker import Chunker
from rag.ingestion.loader import DocumentLoader
from rag.ingestion.models import Document


def test_loader_reads_text_file(tmp_path: Path) -> None:
    file = tmp_path / "doc.txt"
    file.write_text("hello world", encoding="utf-8")

    documents = DocumentLoader().load(file)

    assert len(documents) == 1
    assert documents[0].text == "hello world"
    assert documents[0].source == str(file)
    assert documents[0].metadata["format"] == "txt"


def test_loader_reads_markdown(tmp_path: Path) -> None:
    file = tmp_path / "doc.md"
    file.write_text("# heading\n\nbody", encoding="utf-8")

    documents = DocumentLoader().load(file)

    assert len(documents) == 1
    assert "heading" in documents[0].text
    assert documents[0].metadata["format"] == "md"


def test_loader_walks_directory(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("alpha", encoding="utf-8")
    (tmp_path / "b.md").write_text("beta", encoding="utf-8")
    (tmp_path / "ignored.bin").write_bytes(b"\x00\x01")

    documents = DocumentLoader().load(tmp_path)

    sources = sorted(Path(d.source).name for d in documents)
    assert sources == ["a.txt", "b.md"]


def test_loader_missing_path_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        DocumentLoader().load(tmp_path / "missing.txt")


def test_chunker_produces_overlapping_chunks() -> None:
    text = "ABCDEFGHIJKL"  # 12 chars
    chunker = Chunker(size=5, overlap=2)
    chunks = chunker.chunk([Document(text=text, source="m")])

    assert [c.text for c in chunks] == ["ABCDE", "DEFGH", "GHIJK", "JKL"]
    assert [c.chunk_index for c in chunks] == [0, 1, 2, 3]


def test_chunker_skips_empty_documents() -> None:
    chunker = Chunker(size=10, overlap=2)
    assert chunker.chunk([Document(text="", source="x")]) == []
    assert chunker.chunk([Document(text="   \n", source="x")]) == []


def test_chunker_validates_overlap() -> None:
    with pytest.raises(ValueError):
        Chunker(size=10, overlap=10)
    with pytest.raises(ValueError):
        Chunker(size=0, overlap=0)


def test_chunker_metadata_inherits_from_document() -> None:
    chunker = Chunker(size=20, overlap=5)
    chunks = chunker.chunk([Document(text="x" * 50, source="src", metadata={"author": "alice"})])
    assert all(c.metadata["author"] == "alice" for c in chunks)
    assert all("char_start" in c.metadata for c in chunks)


def test_loader_nested_and_case_insensitive(tmp_path: Path) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "notes.MD").write_text("nested text", encoding="utf-8")
    assert DocumentLoader().load(tmp_path)[0].text == "nested text"


def test_loader_skips_corrupt_pdf(tmp_path: Path) -> None:
    (tmp_path / "bad.pdf").write_bytes(b"not a PDF")
    (tmp_path / "good.txt").write_text("keep me", encoding="utf-8")
    assert [d.text for d in DocumentLoader().load(tmp_path)] == ["keep me"]


def test_loader_official_pdf_has_text() -> None:
    documents = DocumentLoader().load(Path("data/workshop_corpus/rag_paper_summary.pdf"))
    assert len(documents) == 1 and "retrieval" in documents[0].text.lower()
    assert documents[0].metadata["page_count"] > 0


def test_chunker_exact_boundary_has_no_duplicate_tail() -> None:
    chunks = Chunker(size=5, overlap=2).chunk([Document("ABCDE", "s")])
    assert len(chunks) == 1
    assert chunks[0].metadata["char_end"] == 5


def test_chunker_does_not_mutate_document_metadata() -> None:
    doc = Document("A" * 12, "s", {"author": "Ali"})
    chunks = Chunker(size=5, overlap=2).chunk([doc])
    assert doc.metadata == {"author": "Ali"}
    assert all(c.metadata is not doc.metadata for c in chunks)


def test_loader_replaces_invalid_utf8(tmp_path: Path) -> None:
    file = tmp_path / "text.txt"
    file.write_bytes(b"hello\xffworld")
    assert DocumentLoader().load(file)[0].text == "hello\ufffdworld"

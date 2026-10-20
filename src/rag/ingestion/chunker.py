"""Workshop 2: deterministic character chunking with overlap."""

from __future__ import annotations

from dataclasses import replace

from rag.config import settings
from rag.ingestion.models import Chunk, Document


class Chunker:
    """Split Documents into fixed-size, overlapping chunks."""

    def __init__(
        self,
        size: int | None = None,
        overlap: int | None = None,
    ) -> None:
        self.size = size if size is not None else settings.chunk_size
        self.overlap = overlap if overlap is not None else settings.chunk_overlap
        if self.size <= 0:
            raise ValueError("size must be > 0")
        if self.overlap < 0 or self.overlap >= self.size:
            raise ValueError("overlap must satisfy 0 <= overlap < size")

    def chunk(self, documents: list[Document]) -> list[Chunk]:
        """Build a flat list of chunks from all documents."""
        chunks: list[Chunk] = []
        for document in documents:
            chunks.extend(self._chunk_one(document))
        return chunks

    def _chunk_one(self, doc: Document) -> list[Chunk]:
        """Chunk one document; keep its source and metadata."""
        text = doc.text
        if not text.strip():
            return []

        result: list[Chunk] = []
        step = self.size - self.overlap
        for start in range(0, len(text), step):
            end = min(start + self.size, len(text))
            piece = text[start:end].strip()
            if piece:
                index = len(result)
                result.append(
                    Chunk(
                        text=piece,
                        source=doc.source,
                        chunk_index=index,
                        metadata={
                            **doc.metadata,
                            "chunk_index": index,
                            "char_start": start,
                            "char_end": end,
                        },
                    )
                )
            if end >= len(text):
                break
        return result

    @staticmethod
    def renumber(chunks: list[Chunk]) -> list[Chunk]:
        """Renumber chunks globally when combining lists."""
        return [
            replace(c, chunk_index=i, metadata={**c.metadata, "chunk_index": i})
            for i, c in enumerate(chunks)
        ]

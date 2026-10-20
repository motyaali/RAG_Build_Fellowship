"""Document loading from disk: PDF, plain text, and Markdown.

Workshop 2 implementation for the Build Fellowship student-starter scaffold.
"""

from __future__ import annotations

from pathlib import Path

from rag.ingestion.models import Document
from rag.utils.logging import get_logger

log = get_logger(__name__)

_TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".rst"}
_PDF_SUFFIXES = {".pdf"}
_SUPPORTED_SUFFIXES = _TEXT_SUFFIXES | _PDF_SUFFIXES


class DocumentLoader:
    """Loads documents from a single file or recursively from a directory."""

    def load(self, path: Path) -> list[Document]:
        """Load supported documents, skipping unreadable or unsupported files."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)
        if path.is_file():
            doc = self._load_one(path)
            return [doc] if doc is not None else []

        documents: list[Document] = []
        for file in sorted(path.rglob("*")):
            if not file.is_file():
                continue
            if file.suffix.lower() not in _SUPPORTED_SUFFIXES:
                log.warning("Skipping unsupported file: %s", file)
                continue
            doc = self._load_one(file)
            if doc is not None:
                documents.append(doc)
        return documents

    def _load_one(self, path: Path) -> Document | None:
        """Dispatch to the right reader; log and skip a broken file."""
        suffix = path.suffix.lower()
        try:
            if suffix in _PDF_SUFFIXES:
                return self._load_pdf(path)
            if suffix in _TEXT_SUFFIXES:
                return self._load_text(path)
            log.warning("Skipping unsupported file: %s", path)
        except Exception as exc:
            log.warning("Skipping unreadable file %s: %s", path, exc)
        return None

    @staticmethod
    def _load_text(path: Path) -> Document:
        """Read UTF-8 plain text or Markdown into a Document."""
        text = path.read_text(encoding="utf-8", errors="replace")
        return Document(
            text=text,
            source=str(path),
            metadata={
                "format": path.suffix.lower().lstrip("."),
                "size_bytes": path.stat().st_size,
            },
        )

    @staticmethod
    def _load_pdf(path: Path) -> Document:
        """Read a text PDF and separate pages with form feeds."""
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        text = "\n\f\n".join(page.extract_text() or "" for page in reader.pages)
        return Document(
            text=text,
            source=str(path),
            metadata={
                "format": "pdf",
                "page_count": len(reader.pages),
                "size_bytes": path.stat().st_size,
            },
        )

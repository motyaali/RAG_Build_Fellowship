"""Route handlers for the FastAPI app — Workshop 6.

Routes are registered on a ``router`` and included by ``rag.api.main``. Each
handler depends on the singleton :class:`RAGPipeline` constructed in the app's
``lifespan``; we read it back via ``request.app.state``.
"""

# ruff: noqa: F401  -- imports become used once Workshop 6 is implemented.

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Request

from rag.api.schemas import (
    AskRequest,
    AskResponse,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    Source,
)
from rag.config import settings
from rag.ingestion.chunker import Chunker
from rag.ingestion.loader import DocumentLoader

router = APIRouter()


def _state(request: Request):  # type: ignore[no-untyped-def]
    """Return the application state container set up in ``lifespan``."""
    return request.app.state


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    """Lightweight liveness check that also reports current configuration."""
    state = _state(request)
    return HealthResponse(
        status="ok",
        llm_provider=settings.llm_provider,
        embeddings_provider=settings.embeddings_provider,
        indexed_chunks=state.store.count(),
    )


@router.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest, request: Request) -> AskResponse:
    """Run the RAG pipeline against ``payload.question`` and return an answer."""
    pipeline = _state(request).pipeline
    result = pipeline.answer(
        payload.question,
        k=payload.top_k,
        temperature=payload.temperature,
    )
    sources = [
        Source(
            source=chunk.source,
            chunk_index=chunk.chunk_index,
            text=chunk.text,
            metadata=chunk.metadata,
        )
        for chunk in result.sources
    ]
    return AskResponse(
        answer=result.text,
        sources=sources,
        question=result.question,
    )


@router.post("/ingest", response_model=IngestResponse)
def ingest(payload: IngestRequest, request: Request) -> IngestResponse:
    """Re-run ingestion against a directory or file path."""
    target = Path(payload.path).expanduser() if payload.path else settings.data_dir
    target = target.resolve()
    if not target.exists():
        raise HTTPException(status_code=404, detail=f"Ingestion path not found: {target}")
    if not (target.is_file() or target.is_dir()):
        raise HTTPException(status_code=400, detail="Ingestion path must be a file or directory")

    state = _state(request)
    try:
        documents = DocumentLoader().load(target)
        chunks = Chunker().chunk(documents)
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"Could not load ingestion path: {exc}") from exc

    if payload.clear:
        state.store.clear()
    try:
        state.store.add(chunks)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Could not embed or index documents: {exc}") from exc

    return IngestResponse(
        documents_loaded=len(documents),
        chunks_created=len(chunks),
        chunks_indexed=state.store.count(),
        path=str(target),
    )

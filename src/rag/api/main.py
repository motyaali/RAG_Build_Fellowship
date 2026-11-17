"""FastAPI application entrypoint — Workshop 6.

Run locally::

    uv run uvicorn rag.api.main:app --reload

Open http://127.0.0.1:8000/docs for the auto-generated Swagger UI.
"""

# ruff: noqa: F401  -- imports become used once Workshop 6 is implemented.

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from rag.api.routes import router
from rag.generation.pipeline import RAGPipeline
from rag.providers.factory import get_embeddings, get_llm
from rag.retrieval.retriever import Retriever
from rag.utils.logging import get_logger
from rag.vectorstore.chroma_store import ChromaStore

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Construct the pipeline once at startup, tear it down at shutdown."""
    embeddings = get_embeddings()
    llm = get_llm()
    store = ChromaStore(embeddings=embeddings)
    retriever = Retriever(store=store)
    pipeline = RAGPipeline(retriever=retriever, llm=llm)

    app.state.embeddings = embeddings
    app.state.llm = llm
    app.state.store = store
    app.state.retriever = retriever
    app.state.pipeline = pipeline
    log.info("RAG API ready; indexed_chunks=%s", store.count())
    try:
        yield
    finally:
        log.info("RAG API shutting down")


def create_app() -> FastAPI:
    """Build and return the FastAPI application.

    Defined as a factory so tests can build a fresh app per session.
    """
    app = FastAPI(
        title="RAG Build Fellowship API",
        version="0.1.0",
        description="Document-grounded question answering for a local corpus.",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )
    app.include_router(router)
    return app


app: FastAPI = create_app()
"""Module-level app instance used by ``uvicorn rag.api.main:app``."""

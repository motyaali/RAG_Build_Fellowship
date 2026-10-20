# RAG Build Fellowship

Weeks 1–4 implementation for the Build Fellowship project led by Mayank Gowda.

## Setup and checks

```sh
uv sync --locked
uv run python scripts/verify_setup.py --offline
uv run python scripts/ingest.py --dry-run
uv run pytest
uv run ruff check .
```

The default corpus contains seven official documents and produces 47 chunks at chunk size 500 / overlap 50.

For live Gemini setup, copy `.env.example` to `.env`, configure the key and model locally, and run `uv run python scripts/verify_setup.py`. API keys and vector databases are excluded from Git.

## Indexing and retrieval

```sh
uv run python scripts/ingest.py
uv run python scripts/query.py --retrieve-only --top-k 3 "What is RAG?"
```

These commands use the configured embedding provider. A free local validation path is also available:

```sh
uv run python scripts/validate_retrieval.py
```

The local path uses all-MiniLM-L6-v2 (384 dimensions), downloads its public model on first run, and writes `.chroma-local/`. It does not call Gemini. Do not mix embedding models in one database.

Week 4 query and metadata-filter examples are in [notebooks/week4_retrieval.ipynb](notebooks/week4_retrieval.ipynb). Tests cover ingestion, persistent storage, upserts and retrieval.

Live Gemini indexing/querying has not been reverified in this copy. Local semantic retrieval has been verified.

## Attribution

Based on [Mayank Gowda's student starter](https://github.com/mayankgowda/oaf-rag-build-project/tree/student-starter). The seven-document corpus is from instructor main commit `8290704fdd8bb6074cffb74c46304a9377fb376e`. Instructor documentation and later-workshop stubs are preserved. Implementation and tests were prepared with Codex assistance.

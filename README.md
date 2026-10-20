# Build Fellowship — RAG application

Motya Ali's project work through **Week 4: Retrieve Context** (October 20, 2026).

This repository is the central source for the code, official sample corpus, tests, retrieval notebook, and validation evidence. It builds on [Mayank Gowda's instructor repository](https://github.com/mayankgowda/oaf-rag-build-project/tree/student-starter). See [STATUS.md](STATUS.md) for the evidence and remaining checks.

## Weeks 1–4 status

| Week | Deliverable | Verified status |
|---|---|---|
| 1 | Environment and provider setup | Python/uv/dependencies verified locally. Earlier live Gemini success is documented; it has not been repeated in this copy. No API key is stored here. |
| 2 | Document loader and overlapping chunker | Seven official documents → 47 chunks at size 500 / overlap 50; 14 ingestion tests enabled. |
| 3 | Persistent ChromaStore | Initialization, add/upsert, query, count and clear implemented. 47 local semantic vectors; persistence and duplicate prevention verified. Live Gemini indexing remains unverified. |
| 4 | Retriever and similarity search | Ranked retrieval, default/explicit top-k, metadata filters, and executed notebook implemented and verified. |

Later-workshop generation and API stubs are retained from the instructor scaffold and are outside the current deliverable.

[October 20 speaking notes](docs/meeting-notes-oct20.md) are included for the touchbase.

## Review in five minutes

1. Read [the status and evidence](STATUS.md).
2. Inspect [ChromaStore](src/rag/vectorstore/chroma_store.py) and [Retriever](src/rag/retrieval/retriever.py).
3. Open [the executed retrieval notebook](notebooks/week4_retrieval.ipynb).
4. Inspect [eight semantic query results](evidence/local-retrieval.json).
5. Run the checks below.

## Run without API keys

Install Python 3.12 and [uv](https://docs.astral.sh/uv/), then open a terminal in this repository:

```sh
uv sync --locked
uv run python scripts/verify_setup.py --offline
uv run python scripts/ingest.py --dry-run
uv run python scripts/validate_retrieval.py
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

The local semantic validation downloads the public all-MiniLM-L6-v2 ONNX model (~80 MB) on first use. It makes no Gemini or LLM calls. It writes 47 vectors to `.chroma-local/` and checks four queries at both k=3 and k=5. The model cache and database stay local and are excluded from Git.

For ordinary CLI ingestion and retrieval with the local provider, use PowerShell:

```powershell
$env:PYTHONUTF8='1'
$env:EMBEDDINGS_PROVIDER='local_onnx'
$env:CHROMA_PERSIST_DIR='.chroma-local'
uv run python scripts/ingest.py
uv run python scripts/query.py --retrieve-only --top-k 3 "How does ChromaDB persist vectors?"
```

`--retrieve-only` returns chunks without running the future answer-generation module. On macOS/Linux, set the same environment variables with `export`.

## Gemini configuration

Copy `.env.example` to `.env` locally and supply your own free-tier key. Never commit `.env`. Keep the same embedding model for indexing and questions. Use a different persistence directory for each model; local 384-dimensional vectors cannot be mixed with Gemini vectors. Live ingestion makes real provider calls and was not run for this recovered copy. Do not clear any existing database to switch models.

The seven official samples are in `data/workshop_corpus/` and are the default data directory. The original three starter placeholders remain in `data/sample_docs/`. The instructor's main-branch corpus is credited in [STATUS.md](STATUS.md).

## Attribution

Instructor scaffold, protocols, Gemini provider, architecture documents, and official sample documents: Mayank Gowda. Week 2 modules were recovered from Motya Ali's saved project files. Week 3/4 implementation, extra tests, local validation tools, and review materials were prepared with Codex assistance. Original instructor instructions are preserved in [docs/instructor-readme.md](docs/instructor-readme.md).

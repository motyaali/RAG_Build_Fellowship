# Weeks 1–4 status and evidence

Snapshot: October 20, 2026. This is a student project through Week 4, not a finished eight-week application.

## Completed code and checks

- Loader supports text, Markdown and text PDFs, recursive directories, unsupported/corrupt-file skipping, and source metadata.
- Chunker preserves source, chunk index and character offsets; validates overlap; baseline: 7 documents and 47 chunks at 500/50.
- ChromaStore implements persistent collection setup, injected embeddings, upsert, query, count and clear. It reconstructs source/index metadata and handles empty input and oversized k.
- Gemini provider already embeds one string per request; a mock test confirms per-chunk calls. Store and retriever use the same injected provider for indexing and questions.
- Retriever.search supports default/explicit k and metadata filters.
- Local ONNX semantic validation persists 47 vectors, re-ingests without duplicates, reopens the database, and passes eight top-3/top-5 expected-source checks plus filtering. Model: all-MiniLM-L6-v2, 384 dimensions. This is real local semantic inference, not mocked similarity.
- Executed notebook: `notebooks/week4_retrieval.ipynb`. Full query text and sources: `evidence/local-retrieval.json`.
- Test, lint, formatting and CLI outputs are recorded in `evidence/checks.txt` after verification.

## Remaining checks

1. **Live Gemini setup/index/query:** prior Week 1 success was documented in earlier project records, but this copy has no key and has not repeated that call. The local semantic result does not prove Gemini success. Run with a locally configured free-tier key before marking this verified.
2. **Official attendance/submission record:** pre-survey completion was confirmed from the official receipt. Meeting records support Ali's participation, but the official attendance total and weekly submission destination have not been verified. A repository link does not itself prove instructor acceptance or formal submission.
3. **Original earlier clone:** its location was not established. Saved Week 2 modules were recovered without replacing other user files. This repository is the central working version going forward.

## Requirements and provenance

The instructor's October 13 assignment was add/query, persistent indexing, and sensible top-3/top-5 search results. Workshop 4 adds Retriever.search, example queries and metadata filtering in a notebook. Sources:

- [Instructor workshop guide](https://github.com/mayankgowda/oaf-rag-build-project/blob/main/docs/workshop-guide.md)
- [Instructor architecture specification](https://github.com/mayankgowda/oaf-rag-build-project/blob/student-starter/ARCHITECTURE_SPEC.md), section 6

Base scaffold: student-starter commit `fa29f6b96dbbb9ef3c138ba9336592999f3fbd90`.
Seven-document corpus: instructor main commit `8290704fdd8bb6074cffb74c46304a9377fb376e`, copied into `data/workshop_corpus` with original bytes preserved. The shipped sample PDF triggers pypdf repair warnings but extracts successfully; it has not been modified.

The starter explicitly specifies cosine collection distance. This implementation follows it. The meeting recap described L2 distance; confirm the intended teaching choice with Mayank.

The 15 skipped starter tests cover future generation, API, provider extensions and full-pipeline work. They are not counted as passing Week 1–4 verification.

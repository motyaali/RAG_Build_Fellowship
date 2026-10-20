# October 20 touchbase — Weeks 1–4

## Short update

The project is now centralized in this repository. The loader handles text, Markdown and PDFs, and the seven official documents produce 47 overlapping chunks. ChromaStore and Retriever.search are implemented. Local semantic validation persists 47 vectors, prevents duplicate ingestion, and returns the expected source in all eight top-3/top-5 query checks. The notebook demonstrates source filtering. The full local suite passes 27 tests; 15 tests for future-workshop modules remain skipped.

The outstanding technical verification is live Gemini indexing/querying in this recovered copy. Local semantic testing used all-MiniLM-L6-v2, not Gemini. Earlier Week 1 Gemini success was documented, but it has not been repeated here.

## Explain the design

- The loader normalizes supported files into Document objects. The chunker preserves source, index and character offsets.
- The 500-character size / 50-character overlap preserves nearby context at boundaries.
- The store receives one vector per chunk. Stable source/index IDs make repeated upserts safe.
- The same embedding provider is used to index documents and embed questions.
- The retrieval layer supplies a clean place for filters, reranking and later query rewriting.
- Cosine distance follows the instructor's explicit starter-code instruction. The meeting recap mentioned L2; confirm the teaching choice.

## Show Mayank

1. README Weeks 1–4 table and STATUS.md.
2. ChromaStore and Retriever source files.
3. Executed notebooks/week4_retrieval.ipynb.
4. evidence/local-retrieval.json and evidence/checks.txt.
5. GitHub Actions result for the current main commit.

## Confirm during the meeting

- Is sharing this repository the weekly submission process, or is there another submission destination?
- Does the local-model demonstration satisfy this week's check, or should Gemini results also be submitted?
- Confirm the official attendance record and whether any makeup assignment is needed.

The official pre-project survey receipt was verified. This repository contains code and technical evidence; it does not replace the program's official attendance and submission records.

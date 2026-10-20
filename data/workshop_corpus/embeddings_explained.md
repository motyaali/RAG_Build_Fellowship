# Embeddings, plainly explained

An embedding is a list of floating-point numbers that represents the meaning of
a piece of text. Two pieces of text with similar meaning produce vectors that
point in similar directions. Two pieces of text about wholly different topics
produce vectors that point in different directions. That's the whole trick.

## Dimensions

The number of floats in a vector is its *dimension*. Common choices:

- OpenAI `text-embedding-3-small` — 1536 dimensions
- OpenAI `text-embedding-3-large` — 3072 dimensions
- Gemini `gemini-embedding-2` — 3072 dimensions (truncatable via MRL)
- `sentence-transformers/all-MiniLM-L6-v2` — 384 dimensions

Higher dimensions can encode finer semantic distinctions but cost more to
store and search. For most teaching and prototype workloads, 384–768
dimensions are plenty.

A collection in your vector database must be consistent: every vector in the
same collection has to share the same dimensionality. If you swap embedding
models mid-project, delete the collection and re-ingest.

## Similarity metrics

Once you have vectors, you compare them with a *distance* (or its inverse,
*similarity*) function. The three you'll meet most often:

| Metric | What it measures | When to use |
|--------|------------------|-------------|
| Cosine | Angle between vectors, ignores magnitude | Default for modern transformer-based embeddings |
| L2 (Euclidean) | Straight-line distance | OK if your vectors are L2-normalised |
| Dot product | Magnitude × cosine | Faster but assumes normalised vectors |

ChromaDB defaults to L2. For most contemporary embedding models you want
cosine — set it at collection creation time via
`metadata={"hnsw:space": "cosine"}`.

## What embeddings cannot do

- They don't reason. The vector for "Paris is the capital of France" is close
  to the vector for "Paris is not the capital of France." Negation is the
  classic failure mode.
- They don't preserve exact wording. A search for a specific phone number or
  serial number is much better served by full-text search than embeddings.
- They aren't deterministic across model versions. Re-running the same text
  through a "v3" model after using "v2" produces different vectors.

## Practical defaults

For a teaching RAG system on a small corpus:

- Use whatever embedding model your default LLM provider ships (less
  configuration, less chance of dimension mismatch).
- Pick cosine similarity unless you have a reason not to.
- Re-embed the whole corpus whenever you change embedding model. Cheap on
  small corpora; non-negotiable.

When the corpus grows to hundreds of thousands of chunks, profile the embed
step separately from the index step — they have very different cost curves.

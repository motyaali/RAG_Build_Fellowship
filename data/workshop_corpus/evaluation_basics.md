# Evaluating a RAG system

Evaluating RAG is harder than evaluating either of its halves in isolation.
A bad answer can come from bad retrieval, bad generation, or a perfectly
reasonable model hitting a corpus that doesn't contain the answer. The
discipline is mostly about isolating which half is to blame.

## The two axes

Every RAG evaluation eventually splits into two questions:

1. **Did we retrieve the right context?** (a retrieval problem)
2. **Given the retrieved context, did we generate a good answer?** (a
   generation problem)

Treat them separately. If retrieval is broken, no amount of prompt tuning
will fix the answer. If generation is broken, even perfect retrieval looks
useless from the outside.

## Retrieval metrics

If you have a labelled set of (question, relevant-document) pairs:

- **Recall@k** — of the documents you know are relevant, how many appear in
  the top-k retrieved? Higher is better.
- **MRR (mean reciprocal rank)** — averages 1/rank of the first relevant
  result across queries. Punishes burying the right answer at rank 9.
- **nDCG** — like MRR but weights multiple relevant results graded by their
  position.

You almost never have a labelled set on day one. The lightweight substitute
is to write 10–20 questions you know your corpus *should* answer, and
eyeball whether the right chunks are in the top-3.

## Generation metrics

Three rough buckets, in order of how seriously practitioners take them:

- **Human ratings** — slow, expensive, the gold standard. A 5-point scale
  for groundedness and relevance, two raters per item.
- **LLM-as-judge** — use a stronger model to grade a weaker model's answers
  against a rubric. Surprisingly correlated with human judgement on coarse
  scales; deeply unreliable on fine ones.
- **Reference-based scores (BLEU, ROUGE)** — were designed for translation
  and summarisation. They reward token overlap with a reference answer. For
  RAG, they're better than nothing and worse than a careful eyeball.

## What to actually do

For a project of this size, a pragmatic evaluation loop:

1. Keep a `tests/golden_questions.txt` of ~20 questions.
2. After any change to chunking, retrieval, or the prompt, run those
   questions through the pipeline and diff the answers against the previous
   run.
3. Investigate every regression. The answers will be different — most
   differences are improvements, some are bugs.

This catches obvious breakage without the overhead of a formal eval harness.
When you start caring about the size of regressions, add LLM-as-judge to
quantify them.

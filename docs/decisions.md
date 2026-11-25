# Decisions

## Chunk size 800 characters, overlap 150

**Decision:** split each page into chunks of 800 characters with 150 overlap.

**Evidence** (transformer.pdf, 15 questions, top 5, mean reciprocal rank):

| Size / overlap | MRR |
|---|---|
| 300 / 60 | (your number) |
| 500 / 100 | (your number) |
| 650 / 130 | 0.80 |
| 800 / 150 | 0.87 |
| 1000 / 200 | 0.71 |

**Why:** highest MRR. The embedding model reads about 256 word pieces
(roughly 1000 characters), so chunks near that limit lose their tail.

**Limits:** one document and 15 questions, where one question moves MRR by
0.03 to 0.07. Re-check with more documents and after changing the extractor
or the chunker.
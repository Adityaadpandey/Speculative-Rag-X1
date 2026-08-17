# LLM-rerank answer-quality eval — ta_native, 30 validation queries
Paired comparison against embedding-rank-1 baseline, same candidate set per query.

## Answer quality
| metric | baseline (rank-1) | LLM rerank |
|---|---|---|
| mean word-overlap F1 vs gold answer | 0.068 | 0.124 |
| % answer sourced from a gold-relevant passage | 3.3% | 20.0% |

## Rerank call latency (async, NOT counted against the 200ms RAG SLA)
| metric | value |
|---|---|
| P50 (ms) | 4184.0 |
| P70 (ms) | 4833.1 |
| P100 (ms) | 9858.0 |

rerank fixed 5 queries that rank-1 got wrong, and broke 0 queries that rank-1 got right.

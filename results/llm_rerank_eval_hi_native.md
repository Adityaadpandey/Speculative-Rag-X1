# LLM-rerank answer-quality eval — hi_native, 30 validation queries
Paired comparison against embedding-rank-1 baseline, same candidate set per query.

## Answer quality
| metric | baseline (rank-1) | LLM rerank |
|---|---|---|
| mean word-overlap F1 vs gold answer | 0.225 | 0.218 |
| % answer sourced from a gold-relevant passage | 13.3% | 30.0% |

## Rerank call latency (async, NOT counted against the 200ms RAG SLA)
| metric | value |
|---|---|
| P50 (ms) | 3767.2 |
| P70 (ms) | 4553.5 |
| P100 (ms) | 6440.3 |

rerank fixed 7 queries that rank-1 got wrong, and broke 2 queries that rank-1 got right.

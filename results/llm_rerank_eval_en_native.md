# LLM-rerank answer-quality eval — en_native, 30 validation queries
Paired comparison against embedding-rank-1 baseline, same candidate set per query.

## Answer quality
| metric | baseline (rank-1) | LLM rerank |
|---|---|---|
| mean word-overlap F1 vs gold answer | 0.267 | 0.236 |
| % answer sourced from a gold-relevant passage | 23.3% | 30.0% |

## Rerank call latency (async, NOT counted against the 200ms RAG SLA)
| metric | value |
|---|---|
| P50 (ms) | 2937.7 |
| P70 (ms) | 3285.8 |
| P100 (ms) | 5972.1 |

rerank fixed 5 queries that rank-1 got wrong, and broke 3 queries that rank-1 got right.

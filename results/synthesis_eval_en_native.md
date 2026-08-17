# Multi-passage synthesis recovery eval — en_native, 30 sampled queries, 4 were INSUFFICIENT_EVIDENCE

## Recovery
| metric | value |
|---|---|
| INSUFFICIENT_EVIDENCE rows attempted | 4 |
| recovered (synthesis found an answer) | 4 (100.0%) |
| mean word-overlap F1 vs gold answer (recovered rows) | 0.150 |

## Synthesis call latency (async, NOT counted against the 200ms RAG SLA)
| metric | value |
|---|---|
| P50 (ms) | 6071.0 |
| P70 (ms) | 6268.2 |
| P100 (ms) | 7351.7 |

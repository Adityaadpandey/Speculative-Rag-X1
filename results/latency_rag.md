# RAG-pipeline-only latency — `hi_native`, 200 validation queries

| metric | value |
|---|---|
| P50 (ms) | 211.35 |
| P70 (ms) | 215.91 |
| P100 (ms) | 1299.63 |
| target | <200ms |
| P100 within target | NO |

## Guardrail outcome breakdown

| outcome | count |
|---|---|
| answered | 176 |
| off_topic_refused | 0 |
| unsafe_input_refused | 0 |
| ungrounded_refused | 0 |
| insufficient_evidence | 24 |

top-1 retrieval score across scored queries: min=0.999 median=15.499 max=34.060 (off_topic_threshold=1e-06)

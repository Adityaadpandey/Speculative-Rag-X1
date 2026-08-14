# RAG-pipeline-only latency — `hi_native`, 200 validation queries

| metric | value |
|---|---|
| P50 (ms) | 7.16 |
| P70 (ms) | 7.45 |
| P100 (ms) | 43.17 |
| target | <200ms |
| P100 within target | YES |

## Guardrail outcome breakdown

| outcome | count |
|---|---|
| answered | 200 |
| off_topic_refused | 0 |
| unsafe_input_refused | 0 |
| ungrounded_refused | 0 |

top-1 retrieval score across scored queries: min=0.790 median=0.894 max=0.973 (off_topic_threshold=0.75)

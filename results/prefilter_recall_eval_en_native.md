# Prefilter recall eval (Phase 1, offline, no LLM calls) — en_native, 300 queries

Go/no-go gate for config.SETTINGS.rerank_prefilter_enabled: this must show ZERO gold-passage exclusions before the flag can flip to True.

## Results
| metric | value |
|---|---|
| queries with gold passage in the retrieved pool | 11 / 300 |
| gold-passage exclusions caused by the prefilter | 1 |
| mean pool size before / after prefilter | 19.5 / 10.1 |

**FAIL** — the prefilter excluded the gold-relevant passage on 1 query. Do not enable rerank_prefilter_enabled until this is resolved. Excluded query_ids: 1000584

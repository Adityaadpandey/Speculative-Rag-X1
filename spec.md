# Voice-Enabled Indic RAG — Technical Plan

**Assignment reference:** Voice → STT → Chunking/Retrieval (vector DB) → Answer generation
**Dataset:** `ai4bharat/MSMARCO-XI` (IndicRAGSuite, arXiv:2506.01615) — 11,451,314 rows, 55.6GB, 14 Indic languages

---

## 0. What the dataset actually is, and why it changes the plan

This is not a raw document dump — it's **MS MARCO's passage-ranking benchmark, machine-translated into 14 Indian languages** (Assamese, Bengali, Gujarati, Hindi, Kannada, Malayalam, Marathi, Nepali, Odia, Punjabi, Sanskrit, Tamil, Telugu, Urdu). Every row already contains:

- `query` (translated) / `Eng_Query` (original English)
- `Answer` (translated) / `Eng_Answer` (original English)
- `passages.English_passages` / `passages.Translated_passages` — ~10 candidate passages per query
- `passages.is_selected` — **gold relevance labels** (1 = actually relevant, 0 = distractor)
- `query_type` — category metadata (DESCRIPTION, NUMERIC, etc.)

Three consequences that change your build, not just your understanding:

1. **You get a free, gold-labeled retrieval eval set.** `is_selected` is exactly the ground truth you'd otherwise have to build by hand. Use it to compute Recall@K / MRR for whatever chunking and retrieval strategy you pick — this is your strongest evidence for requirement #2 ("real thought put into chunking/indexing/retrieval"), and it costs you nothing to compute.
2. **Don't index all 55.6GB.** Passages repeat heavily across queries (classic MS MARCO structure — many queries share source passages), and 14 languages × millions of rows is not a scoped-down demo, it's a multi-day ingestion job on its own. Scope to 2-3 languages deliberately (see §1) and deduplicate passages before indexing.
3. **This is an Indic-language-first task by construction.** That should directly drive your STT provider choice, not be an afterthought (see §2).

---

## 1. Scoping decision: which languages, how much data

**Recommendation: Hindi + English as the core demo, Tamil or Bengali as a "yes, it's actually multilingual" proof point.** Reasoning:

- Hindi + English covers the largest population and gives you the most STT/embedding-model support to draw on
- A third, script-different language (Tamil/Bengali, not another Devanagari-script language) demonstrates the pipeline isn't secretly hardcoded to one script or tokenizer — a reviewer will notice if you only prove it on Hindi
- Full 14-language coverage is a legitimate stretch goal to mention as "supported by design, demoed on 3" rather than something to actually ingest end-to-end under time pressure

**Deduplicate before indexing.** Build your actual vector-store corpus as the *unique* passage set across your chosen languages' rows, not one point per (query, passage) pair — indexing every occurrence multiplies your corpus size for zero retrieval benefit and directly hurts your <200ms retrieval target.

---

## 2. STT provider: Sarvam vs. ElevenLabs, for *this* dataset specifically

| | Sarvam (Saaras v3) | ElevenLabs (Scribe v2 Realtime) |
|---|---|---|
| Indian language coverage | 22 scheduled languages — covers all 14 in this dataset, including lower-resource ones (Assamese, Sanskrit, Nepali, Odia) | 90+ languages globally, ~11 Indian languages spotlighted (Hindi, Tamil, Telugu, Bengali, Malayalam, Gujarati, Kannada, Odia, Marathi, Punjabi, Sindhi) — doesn't clearly cover Assamese, Sanskrit, Nepali |
| Latency | <150ms time-to-first-token (streaming, "Fast" mode) | <150ms (streaming, FLEURS benchmark) |
| Code-mixing (Hinglish/Tanglish) | Purpose-built for this — explicit training focus | Supported, less of a specific focus |
| Fit for this assignment | Built specifically for the Indian-language, code-mixed speech patterns this dataset represents | Excellent general-purpose option, best if you scope to the 11 well-covered languages only |

**Recommendation: Sarvam**, specifically because your scoping choice in §1 might extend to a lower-resource language later, and Sarvam is the only one of the two with clear, dedicated coverage of the dataset's full language list. If you scope strictly to Hindi/English/Tamil/Bengali/Telugu, ElevenLabs is an equally defensible pick and arguably has the edge on general production polish (India data residency, enterprise SLAs) — either choice is justifiable in your writeup, just state the reasoning, since the assignment explicitly wants you to pick one.

---

## 3. The 200ms latency target — clarified: RAG pipeline only, STT excluded

**Confirmed scope:** the 200ms budget covers the RAG pipeline — query embedding, vector DB retrieval, and generation through to final output. STT is timed and reported separately, outside this SLA. That's a real, hittable target, but it's still tight enough that it dictates specific architecture choices, not just tuning:

| Stage | Rough budget | Feasibility |
|---|---|---|
| Query embedding | ~5-15ms | Easy — small local embedding model |
| Vector DB search | ~20-50ms | Easy — Qdrant, quantized vectors, warm index |
| Hot-path guardrail checks | ~0-10ms | Only if heuristic — see §6, no model call here |
| Answer generation | ~100-150ms of what's left | The hard constraint — see below |
| **Total** | **<200ms** | |

**The generation step is where this budget is won or lost, and it rules out calling any hosted LLM API in the timed path.** A network round trip to Claude, OpenAI, or even a cloud-hosted small model typically costs 100-300ms on its own, before a single token is generated — that alone can consume the entire remaining budget, every time, regardless of how good the model is. Concretely:

- **Generation runs on a local/on-device model, not a hosted API call**, if the SLA needs to hold under real measurement rather than a best-case demo run.
- **Favor extractive or near-extractive answer selection as the primary path**, not free-form generation: rank and return the best-matching passage (or a short span within it) rather than having a model compose a new sentence. This is a ranking operation, not a generation one — it runs in single-digit milliseconds and reliably fits the budget, which matters more here than a marginally more fluent answer.
- If you want a more natural, synthesized answer on top of that, treat it as a **secondary, asynchronous enhancement**: return the extractive answer within the 200ms SLA, then optionally stream or append a rephrased version from a small local model afterward — reported as its own latency number, not folded into the hard target.
- If you do put a small generative model in the hot path anyway, it needs to be genuinely tiny (sub-1B to ~1-2B params), quantized, capped to a short `max_tokens`, and ideally served with something optimized for low first-token latency (vLLM, TensorRT-LLM). Benchmark it before committing to it as your primary path — don't assume it fits, measure that it does.

**Concrete techniques for the retrieval leg specifically:**
- Small, fast embedding model (bge-small or an Indic-tuned equivalent — avoid large embedding models here, they're the wrong latency trade for this leg)
- Quantized vectors in Qdrant (int8 or binary quantization) — meaningful speedup at negligible recall cost at this scale
- Keep the index warm/in-memory rather than cold-loading per query
- Tune HNSW `ef_search` down for speed, and measure the recall cost of doing so (report both, don't just pick "fast" blind)
- Keep any re-ranking step (if you add one) outside the hot path, or make it optional/async — a cross-encoder re-ranker alone can blow the whole budget

---

## 4. Chunking strategy — satisfying "vast," honestly

MS MARCO passages are already short, atomic units (~50-100 words) — so naive fixed-size chunking isn't really the failure mode here the way it would be on long documents. The real opportunity to show depth is comparing **multiple retrieval-unit strategies against the gold `is_selected` labels** and reporting which wins, where, and why:

1. **Native passage-level** (use MS MARCO passages as-is) — your baseline/control
2. **Sliding-window fixed-size with overlap** — the "classic" approach, included deliberately as a comparison point, not as your only strategy
3. **Sentence-boundary-aware chunking** with overlap, for anywhere you have longer source text than a single passage
4. **Metadata-aware chunking** — tag every indexed chunk with `language`, `query_type`, and source passage ID, enabling filtered retrieval (e.g., restrict search to one language, or boost/filter by `query_type` when the transcribed query's intent is classifiable)
5. **Hierarchical (parent-child)**, optional stretch — small chunks for precise vector matching, linked back to the full passage for the generation step to read

**Run Recall@K for each strategy against `is_selected`** on a held-out sample of your scoped languages. This is the single most convincing artifact you can put in a writeup for requirement #2 — a table showing which chunking approach actually retrieves the gold passage more often, with real numbers, not just a description of the approaches.

---

## 5. Harness (requirement #5)

Structure the pipeline as explicit, typed stages rather than one function that does everything:

- Each stage (STT, guardrail-check, retrieval, generation, groundedness-check) as its own function/node with a defined input/output schema (Pydantic models are a good fit)
- Retries with exponential backoff on the two network-dependent legs (STT API, LLM API) — these are exactly where transient failures happen
- Per-stage timeouts, tuned tighter than you'd normally use given the latency budget in §3
- Structured logging of every stage's input, output, and timing — this same instrumentation is what feeds your P50/P70/P100 numbers in §7, so build it once and get both requirements from it
- If you want to reuse patterns you already trust: this maps cleanly onto a LangGraph graph — each stage a node, explicit state passed between them, conditional edges for the two guardrail exit points below

## 6. Guardrails (requirement #6) — now budget-constrained too

§3 has a direct consequence here: anything sitting in the timed 200ms path has to be **heuristic, not a model call** — including the groundedness check, which is the one place it's tempting to reach for an LLM.

**In the hot path (must fit inside the budget):**
- **Off-topic detection:** reuse the top-K similarity score you already computed during retrieval — if the best match is below a threshold, the corpus doesn't cover this query. This is effectively free; don't recompute a separate check.
- **Groundedness check:** lexical or embedding overlap between the selected answer and its source passage (cosine similarity, or n-gram overlap) — a few milliseconds, no model call. This replaces an LLM-based check as the thing your hard SLA is measured against.
- **Unsafe-input check:** a lightweight local classifier or keyword/regex-based filter — needs to be fast enough to actually benchmark inside budget, not a call to an external moderation API.

**Outside the hot path (valuable, but reported separately, not counted against the 200ms SLA):**
- A richer LLM-based groundedness check — this is where the Speculative RAG paper's `ρ_Self-reflect` mechanism belongs: *"Does the retrieved passage actually support this generated answer? Yes or no,"* scored as a single verification call. It's a genuinely stronger check than the heuristic version above, it's just not a single-digit-millisecond operation. Run it as an async second pass on top of the extractive fast path, log its latency separately, and be explicit in your writeup that it's a quality layer running outside the timed SLA, not inside it.

This split is a stronger design than cramming everything into one number — it shows you can tell the difference between a check that's cheap-and-necessary versus one that's rich-and-valuable, and that you're not quietly hiding the second kind inside a number it doesn't belong in.

Both the hot-path checks should produce an explicit refusal path, not a silent fallback — the assignment is specifically evaluating whether your system knows its own limits.

---

## 7. Latency analytics (requirement #4)

- Sample 200-500 queries from the dataset's **validation split** (not train — keeps this cleanly separated from anything used to tune chunking/retrieval choices)
- Run the full harness per query, logging per-stage and total latency from the instrumentation in §5
- Report **P50 / P70 / P100** across three separate numbers, not one blended figure:
  1. **RAG pipeline latency** (embedding + retrieval + hot-path guardrails + generation) — this is the one measured against the 200ms target
  2. **STT latency** (Sarvam/ElevenLabs) — reported for completeness, explicitly outside the SLA
  3. **Async groundedness-check latency** (if you implement the LLM-based version from §6) — reported to show you measured it, explicitly not counted against the target
- Note on P100 specifically: it's the single worst observed value, so it's genuinely sensitive to one bad network blip — report it as required, but don't over-index on a single outlier when reasoning about it in your writeup; P50/P70 are the numbers that describe typical behavior

---

## 8. Suggested build order

1. **Data prep** — pull the scoped language subset(s), deduplicate passages, hold out the validation split for eval
2. **Indexing** — implement the chunking strategies from §4, embed, index into Qdrant, run the Recall@K comparison
3. **STT integration** — wire up Sarvam (or ElevenLabs), test against real spoken audio in your target languages
4. **Retrieval + hot-path guardrails** — get embedding + retrieval + heuristic off-topic/groundedness checks under budget per §3/§6; benchmark as you go, not at the end
5. **Extractive generation** — the primary answer path: rank/select the best passage or span, no model call, fits comfortably inside what's left of the 200ms budget
6. **Optional: async LLM enhancement** — a small local model rephrasing the extractive answer, and/or the richer `ρ_Self-reflect`-style groundedness check, both running outside the timed SLA and logged separately
7. **Harness + instrumentation** — wrap it all in the structured pipeline from §5, with the logging that feeds §7
8. **Latency analytics + writeup** — run the 200-500 query eval, produce the three-way P50/P70/P100 tables (RAG SLA / STT / async check), write up the chunking comparison

---

## Appendix: sources

- Wang et al., *Speculative RAG: Enhancing Retrieval Augmented Generation through Drafting*, arXiv:2407.08223 — source of the groundedness-guardrail design in §6
- Pasunuri et al., *IndicRAGSuite: Large-Scale Datasets and a Benchmark for Indian Language RAG Systems*, arXiv:2506.01615 — source dataset paper for `ai4bharat/MSMARCO-XI`

"""Builds the real production hi_native/en_native/ur_native collections from
much larger slices of MS MARCO-XI than the 300-query eval sample the other
scripts use -- this is content to serve, not a benchmark. Hindi and English
share the Hindi-sourced files; Urdu is read from its own Urdu files.

Sized for Qdrant Cloud's free tier (4GB disk, shared across every collection
in the cluster): SAMPLE_QUERIES=40_000 works out to roughly 2.64KB/point for
hi and 1.64KB/point for en (measured from the existing 2,994-point
collections), with about 9.87 unique passages per query, so roughly 1.7GB
total before the Urdu collection -- comfortable headroom under the 4GB cap.
Raise SAMPLE_QUERIES only after redoing that math yourself; there's no
per-collection safety net here.

Unlike the eval scripts, this does not filter to queries with a labeled gold
passage (load_validation_sample does that; real content to index doesn't
need one) and deletes and replaces the existing hi_native, en_native, and
ur_native collections (SparseChunkIndex's constructor semantics), matching
the strategy name the webapp already reads by default.

Usage: uv run python -m scripts.ingest_production_corpus [sample_queries] [language ...]
       Add one or more languages (for example, `40000 ur`) to ingest only
       those collections; omitting them rebuilds hi, en, and ur.
"""

import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import pyarrow.parquet as pq

from speculative_rag.chunking import chunk_native
from speculative_rag.config import REPO_ROOT, SETTINGS, get_qdrant_client
from speculative_rag.data import dedupe_passages
from speculative_rag.indexing import SparseChunkIndex
from speculative_rag.sparse_embedding import SparseEmbedder

SAMPLE_QUERIES = 40_000
# Kept well under what a single upsert can push through within the client's
# timeout against a free-tier cloud cluster (get_qdrant_client sets 60s, but
# 5_000 points/call -- full sparse vectors + text payloads -- still timed
# out against the eu-central-1 free tier; 500 comfortably clears it).
UPSERT_BATCH_SIZE = 500
UPSERT_WORKERS = 8
UPSERT_RETRIES = 4

_HI_TRAIN = REPO_ROOT / "MSMARCO-XI" / "train" / "hintrain.parquet"
_HI_VAL = REPO_ROOT / "MSMARCO-XI" / "validation" / "hinval.parquet"
_UR_TRAIN = REPO_ROOT / "MSMARCO-XI" / "train" / "urdtrain.parquet"
_UR_VAL = REPO_ROOT / "MSMARCO-XI" / "validation" / "urdval.parquet"

# A target is (language, text_field). Hindi and English use the Hindi corpus;
# Urdu uses the Urdu corpus. Urdu's standard language code is "ur".
_TARGETS = [
    ("hi", "Translated_passages"),
    ("en", "English_passages"),
    ("ur", "Translated_passages"),
]
_CORPUS_PATHS = {
    "hi": (_HI_TRAIN, _HI_VAL),
    "ur": (_UR_TRAIN, _UR_VAL),
}
_TARGET_CORPUS = {"hi": "hi", "en": "hi", "ur": "ur"}


_TRAIN_READ_BATCH_SIZE = 5_000


def _read_train_parquet(path) -> pd.DataFrame:
    """Plain pd.read_parquet(path) on hintrain.parquet raises
    ArrowNotImplementedError ("Nested data conversions not implemented for
    chunked array outputs") -- the "passages" struct column's underlying
    string buffers exceed Arrow's single-chunk conversion limit on a file
    this size (hinval.parquet, about 9x smaller, reads fine directly).
    Reading in smaller batches via iter_batches keeps each chunk under that
    limit."""
    batches = list(pq.ParquetFile(path).iter_batches(batch_size=_TRAIN_READ_BATCH_SIZE))
    return pd.concat([b.to_pandas() for b in batches], ignore_index=True)


def load_combined_sample(sample_size: int, seed: int, corpus_language: str = "hi") -> pd.DataFrame:
    """Load a sampled train+validation corpus for Hindi or Urdu."""
    try:
        train_path, validation_path = _CORPUS_PATHS[corpus_language]
    except KeyError as exc:
        raise ValueError(f"Unsupported corpus language: {corpus_language}") from exc

    # The train file is optional: hinval alone has ~98k queries, more than
    # the 40k default sample, so a validation-only checkout still works.
    frames = [pd.read_parquet(validation_path)]
    if train_path.exists():
        frames.insert(0, _read_train_parquet(train_path))
    else:
        print(f"  {train_path.name} not found, sampling from {validation_path.name} only.")
    combined = pd.concat(frames, ignore_index=True)
    n = min(sample_size, len(combined))
    return combined.sample(n=n, random_state=seed).reset_index(drop=True)


def ingest_language(language: str, text_field: str, df: pd.DataFrame, client) -> int:
    passages = list(dedupe_passages(df, language, text_field).values())
    chunks = [c for passage in passages for c in chunk_native(passage)]

    embedder = SparseEmbedder(language)
    collection_name = f"{language}_native"
    index = SparseChunkIndex(client, collection_name)

    # Ingestion is bound by the upsert round trip, not by embedding (the
    # process sat at ~6% CPU doing one batch at a time), so we keep several
    # batches in flight at once. 8 workers took a free-tier cluster from
    # ~700 to several thousand points/s without timeouts.
    def embed_and_upsert(batch):
        vectors = embedder.embed_passages([c.text for c in batch])
        # One SSL handshake timeout used to kill a whole multi-minute ingest.
        # Point ids are uuid5(chunk_id), so re-sending a batch is harmless.
        for attempt in range(UPSERT_RETRIES):
            try:
                index.add(batch, vectors)
                return len(batch)
            except Exception as exc:
                if attempt == UPSERT_RETRIES - 1:
                    raise
                print(f"\n  [{language}] upsert failed ({exc!r:.80}), retrying...")
                time.sleep(2**attempt)

    batches = [chunks[s : s + UPSERT_BATCH_SIZE] for s in range(0, len(chunks), UPSERT_BATCH_SIZE)]
    done = 0
    with ThreadPoolExecutor(max_workers=UPSERT_WORKERS) as pool:
        for future in as_completed(pool.submit(embed_and_upsert, b) for b in batches):
            done += future.result()
            print(f"  [{language}] {done}/{len(chunks)} chunks indexed", end="\r")
    print()
    return len(chunks)


def main() -> None:
    arguments = sys.argv[1:]
    sample_size = SAMPLE_QUERIES
    if arguments:
        try:
            sample_size = int(arguments[0])
            arguments = arguments[1:]
        except ValueError:
            pass
    requested_languages = set(arguments)
    known_languages = {language for language, _ in _TARGETS}
    unknown_languages = requested_languages - known_languages
    if unknown_languages:
        raise SystemExit(
            f"Unknown language(s): {', '.join(sorted(unknown_languages))}. "
            f"Choose from: {', '.join(sorted(known_languages))}"
        )
    targets = [
        target for target in _TARGETS
        if not requested_languages or target[0] in requested_languages
    ]

    corpora: dict[str, pd.DataFrame] = {}
    selected_corpora = dict.fromkeys(_TARGET_CORPUS[language] for language, _ in targets)
    for corpus_language in selected_corpora:
        train_path, validation_path = _CORPUS_PATHS[corpus_language]
        print(
            f"Loading {sample_size} queries from {train_path.name} + "
            f"{validation_path.name} (seed={SETTINGS.sample_seed})..."
        )
        corpora[corpus_language] = load_combined_sample(
            sample_size, SETTINGS.sample_seed, corpus_language
        )
        print(f"Loaded {len(corpora[corpus_language])} query rows.")

    client = get_qdrant_client()
    for language, text_field in targets:
        df = corpora[_TARGET_CORPUS[language]]
        print(f"Ingesting '{language}_native' ({text_field})...")
        n_chunks = ingest_language(language, text_field, df, client)
        print(f"  '{language}_native': {n_chunks} chunks indexed.\n")

    ingested = ", ".join(f"{language}_native" for language, _ in targets)
    print(f"Done. {ingested} now serve the sampled corpora.")


if __name__ == "__main__":
    main()

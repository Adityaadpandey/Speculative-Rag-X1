"""Builds the real production hi_native/en_native collections from a much
larger slice of MS MARCO-XI than the 300-query eval sample the other scripts
use -- this is content to serve, not a benchmark. Sources hi/en from both
hintrain.parquet and hinval.parquet (same schema, concatenated), since
English_passages/Translated_passages/query/Eng_Query all live in the
Hindi-sourced files per config.LANGUAGE_SOURCES.

Sized for Qdrant Cloud's free tier (4GB disk, shared across every collection
in the cluster): SAMPLE_QUERIES=40_000 works out to roughly 2.64KB/point for
hi and 1.64KB/point for en (measured from the existing 2,994-point
collections), with about 9.87 unique passages per query, so roughly 1.7GB
total -- comfortable headroom under the 4GB cap. Raise SAMPLE_QUERIES only
after redoing that math yourself; there's no per-collection safety net here.

Unlike the eval scripts, this does not filter to queries with a labeled gold
passage (load_validation_sample does that; real content to index doesn't
need one) and deletes and replaces the existing hi_native/en_native
collections (SparseChunkIndex's constructor semantics), matching the
strategy name the webapp already reads by default.

Usage: uv run python -m scripts.ingest_production_corpus [sample_queries]
"""

import sys

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

_HI_TRAIN = REPO_ROOT / "MSMARCO-XI" / "train" / "hintrain.parquet"
_HI_VAL = REPO_ROOT / "MSMARCO-XI" / "validation" / "hinval.parquet"

# (language, text_field) -- both read from the same hi-sourced parquet files,
# using the same passage_id_for(language, text) identity space the eval scripts use.
_TARGETS = [("hi", "Translated_passages"), ("en", "English_passages")]


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


def load_combined_sample(sample_size: int, seed: int) -> pd.DataFrame:
    train_df = _read_train_parquet(_HI_TRAIN)
    val_df = pd.read_parquet(_HI_VAL)
    combined = pd.concat([train_df, val_df], ignore_index=True)
    n = min(sample_size, len(combined))
    return combined.sample(n=n, random_state=seed).reset_index(drop=True)


def ingest_language(language: str, text_field: str, df: pd.DataFrame, client) -> int:
    passages = list(dedupe_passages(df, language, text_field).values())
    chunks = [c for passage in passages for c in chunk_native(passage)]

    embedder = SparseEmbedder(language)
    collection_name = f"{language}_native"
    index = SparseChunkIndex(client, collection_name)

    for start in range(0, len(chunks), UPSERT_BATCH_SIZE):
        batch = chunks[start : start + UPSERT_BATCH_SIZE]
        vectors = embedder.embed_passages([c.text for c in batch])
        index.add(batch, vectors)
        print(f"  [{language}] {min(start + UPSERT_BATCH_SIZE, len(chunks))}/{len(chunks)} chunks indexed", end="\r")
    print()
    return len(chunks)


def main() -> None:
    sample_size = int(sys.argv[1]) if len(sys.argv) > 1 else SAMPLE_QUERIES

    print(f"Loading {sample_size} queries from hintrain.parquet + hinval.parquet (seed={SETTINGS.sample_seed})...")
    df = load_combined_sample(sample_size, SETTINGS.sample_seed)
    print(f"Loaded {len(df)} query rows.")

    client = get_qdrant_client()
    for language, text_field in _TARGETS:
        print(f"Ingesting '{language}_native' ({text_field})...")
        n_chunks = ingest_language(language, text_field, df, client)
        print(f"  '{language}_native': {n_chunks} chunks indexed.\n")

    print("Done. hi_native and en_native now serve the full sampled corpus.")


if __name__ == "__main__":
    main()

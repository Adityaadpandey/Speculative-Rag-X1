"""Embedding-model A/B eval: does swapping SETTINGS.embedding_model
(multilingual-e5-small) for a larger multilingual model raise recall@k
enough to justify it, before touching the hot-path config?

Writes to throwaway collections named f"{language}_native_{model_suffix}" --
never the production f"{language}_native" collections the webapp and the
other eval scripts read. ChunkIndex deletes and recreates by collection
name, so reusing the production name here would silently wipe out the
existing index. Scoped to the native strategy only: chunking_comparison.md
already showed sliding_window/sentence never beat native, so there's no
reason to burn embedding time re-testing them here.

Usage:
    uv run python -m scripts.run_embedding_model_eval [candidate_model]
Defaults to comparing SETTINGS.embedding_model (baseline) against
intfloat/multilingual-e5-base (candidate).
"""

import sys

from qdrant_client import QdrantClient

from speculative_rag.chunking import chunk_native
from speculative_rag.config import LANGUAGE_SOURCES, SETTINGS
from speculative_rag.data import build_query_records, dedupe_passages, load_validation_sample
from speculative_rag.embedding import Embedder
from speculative_rag.eval import compute_recall_at_k, recall_hit
from speculative_rag.indexing import ChunkIndex

OVERFETCH_MULTIPLIER = 5
DEFAULT_CANDIDATE_MODEL = "intfloat/multilingual-e5-base"


def _model_suffix(model_name: str) -> str:
    return model_name.rsplit("/", 1)[-1].replace("-", "_")


def run_model_eval(model_name: str, qdrant_client: QdrantClient) -> list[dict]:
    embedder = Embedder(model_name=model_name)
    suffix = _model_suffix(model_name)
    results: list[dict] = []
    fetch_limit = max(SETTINGS.top_k_values) * OVERFETCH_MULTIPLIER

    for source in LANGUAGE_SOURCES:
        df = load_validation_sample(source.parquet_path, SETTINGS.sample_size, SETTINGS.sample_seed)
        query_records = build_query_records(df, source.language, source.text_field, source.query_field)
        passages = list(dedupe_passages(df, source.language, source.text_field).values())

        chunks = [c for passage in passages for c in chunk_native(passage)]
        vectors = embedder.embed_passages([c.text for c in chunks])

        collection_name = f"{source.language}_native_{suffix}"
        index = ChunkIndex(qdrant_client, collection_name, vector_size=embedder.dimension)
        index.add(chunks, vectors)

        hits_by_k: dict[int, list[bool]] = {k: [] for k in SETTINGS.top_k_values}
        for record in query_records:
            if not record.relevant_passage_ids:
                continue
            query_vector = embedder.embed_query(record.query_text)
            retrieved = index.search(query_vector, k=fetch_limit)
            for k in SETTINGS.top_k_values:
                hits_by_k[k].append(recall_hit(retrieved, set(record.relevant_passage_ids), k=k))

        for k in SETTINGS.top_k_values:
            results.append(
                {
                    "model": model_name,
                    "dimension": embedder.dimension,
                    "language": source.language,
                    "k": k,
                    "recall": compute_recall_at_k(hits_by_k[k]),
                    "num_queries": len(hits_by_k[k]),
                    "num_chunks": len(chunks),
                }
            )
    return results


def render_markdown_table(results: list[dict]) -> str:
    header = "| model | dim | language | k | recall@k | num_queries | num_chunks |\n"
    header += "|---|---|---|---|---|---|---|\n"
    rows = "\n".join(
        f"| {r['model']} | {r['dimension']} | {r['language']} | {r['k']} | "
        f"{r['recall']:.3f} | {r['num_queries']} | {r['num_chunks']} |"
        for r in results
    )
    return header + rows + "\n"


def main() -> None:
    candidate_model = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CANDIDATE_MODEL
    qdrant_client = QdrantClient(path=str(SETTINGS.qdrant_path))

    print(f"Evaluating baseline model '{SETTINGS.embedding_model}'...")
    baseline_results = run_model_eval(SETTINGS.embedding_model, qdrant_client)

    print(f"Evaluating candidate model '{candidate_model}'...")
    candidate_results = run_model_eval(candidate_model, qdrant_client)

    all_results = baseline_results + candidate_results
    SETTINGS.results_dir.mkdir(parents=True, exist_ok=True)
    output_path = SETTINGS.results_dir / "embedding_model_comparison.md"
    output_path.write_text(render_markdown_table(all_results))
    print(render_markdown_table(all_results))
    print(f"Saved to {output_path}")
    print(
        "\nThrowaway collections written as f'{language}_native_{model_suffix}' -- "
        "production f'{language}_native' collections were not touched."
    )


if __name__ == "__main__":
    main()

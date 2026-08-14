from functools import partial

from qdrant_client import QdrantClient

from speculative_rag.chunking import chunk_metadata_aware, chunk_native, chunk_sentence_aware, chunk_sliding_window
from speculative_rag.config import LANGUAGE_SOURCES, SETTINGS, LanguageSource, Settings
from speculative_rag.data import build_query_records, dedupe_passages, load_validation_sample
from speculative_rag.embedding import Embedder
from speculative_rag.eval import compute_recall_at_k, recall_hit
from speculative_rag.indexing import ChunkIndex

# How many candidate chunks to fetch per query, as a multiple of the largest
# requested k. Multi-chunk strategies can emit several chunks per passage, so
# fetching only k chunks can under-count distinct passages recall_hit needs
# to see; over-fetching gives it enough candidates to scan past duplicates.
OVERFETCH_MULTIPLIER = 5


def _build_chunkers(settings: Settings):
    return {
        "native": chunk_native,
        "sliding_window": partial(
            chunk_sliding_window,
            window_words=settings.sliding_window_words,
            overlap_words=settings.sliding_window_overlap,
        ),
        "sentence": partial(
            chunk_sentence_aware,
            max_words=settings.sentence_max_words,
            overlap_sentences=settings.sentence_overlap_sentences,
        ),
        "metadata": chunk_metadata_aware,
    }


def run_comparison(
    language_sources: list[LanguageSource],
    embedder,
    qdrant_client: QdrantClient,
    settings: Settings,
) -> list[dict]:
    results: list[dict] = []
    chunkers = _build_chunkers(settings)
    fetch_limit = max(settings.top_k_values) * OVERFETCH_MULTIPLIER

    for source in language_sources:
        df = load_validation_sample(source.parquet_path, settings.sample_size, settings.sample_seed)
        query_records = build_query_records(df, source.language, source.text_field)
        passages = list(dedupe_passages(df, source.language, source.text_field).values())

        for strategy_name, chunker in chunkers.items():
            chunks = [c for passage in passages for c in chunker(passage)]
            vectors = embedder.embed_passages([c.text for c in chunks])

            collection_name = f"{source.language}_{strategy_name}"
            index = ChunkIndex(qdrant_client, collection_name, vector_size=embedder.dimension)
            index.add(chunks, vectors)

            hits_by_k: dict[int, list[bool]] = {k: [] for k in settings.top_k_values}
            for record in query_records:
                if not record.relevant_passage_ids:
                    continue
                query_vector = embedder.embed_query(record.query_text)
                retrieved = index.search(query_vector, k=fetch_limit)
                for k in settings.top_k_values:
                    hits_by_k[k].append(recall_hit(retrieved, set(record.relevant_passage_ids), k=k))

            for k in settings.top_k_values:
                results.append(
                    {
                        "language": source.language,
                        "strategy": strategy_name,
                        "k": k,
                        "recall": compute_recall_at_k(hits_by_k[k]),
                        "num_queries": len(hits_by_k[k]),
                        "num_chunks": len(chunks),
                    }
                )

    return results


def render_markdown_table(results: list[dict]) -> str:
    header = "| language | strategy | k | recall@k | num_queries | num_chunks |\n"
    header += "|---|---|---|---|---|---|\n"
    rows = "\n".join(
        f"| {r['language']} | {r['strategy']} | {r['k']} | {r['recall']:.3f} | {r['num_queries']} | {r['num_chunks']} |"
        for r in results
    )
    return header + rows + "\n"


def main() -> None:
    embedder = Embedder()
    qdrant_client = QdrantClient(path=str(SETTINGS.qdrant_path))

    results = run_comparison(LANGUAGE_SOURCES, embedder, qdrant_client, SETTINGS)

    SETTINGS.results_dir.mkdir(parents=True, exist_ok=True)
    output_path = SETTINGS.results_dir / "chunking_comparison.md"
    output_path.write_text(render_markdown_table(results))
    print(render_markdown_table(results))
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()

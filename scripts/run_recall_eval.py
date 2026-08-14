from pathlib import Path

from qdrant_client import QdrantClient

from speculative_rag.chunking import CHUNKERS
from speculative_rag.config import LANGUAGE_SOURCES, SETTINGS, LanguageSource, Settings
from speculative_rag.data import build_query_records, dedupe_passages, load_validation_sample
from speculative_rag.embedding import Embedder
from speculative_rag.eval import compute_recall_at_k, recall_hit
from speculative_rag.indexing import ChunkIndex


def run_comparison(
    language_sources: list[LanguageSource],
    embedder,
    qdrant_client: QdrantClient,
    settings: Settings,
) -> list[dict]:
    results: list[dict] = []

    for source in language_sources:
        df = load_validation_sample(source.parquet_path, settings.sample_size, settings.sample_seed)
        query_records = build_query_records(df, source.language, source.text_field)
        passages = list(dedupe_passages(df, source.language, source.text_field).values())

        for strategy_name, chunker in CHUNKERS.items():
            chunks = [c for passage in passages for c in chunker(passage)]
            vectors = embedder.embed_passages([c.text for c in chunks])

            collection_name = f"{source.language}_{strategy_name}"
            index = ChunkIndex(qdrant_client, collection_name, vector_size=embedder.dimension)
            index.add(chunks, vectors)

            for k in settings.top_k_values:
                hits = []
                for record in query_records:
                    if not record.relevant_passage_ids:
                        continue
                    query_vector = embedder.embed_query(record.query_text)
                    retrieved = index.search(query_vector, k=k)
                    hits.append(recall_hit(retrieved, set(record.relevant_passage_ids), k=k))

                results.append(
                    {
                        "language": source.language,
                        "strategy": strategy_name,
                        "k": k,
                        "recall": compute_recall_at_k(hits),
                        "num_queries": len(hits),
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

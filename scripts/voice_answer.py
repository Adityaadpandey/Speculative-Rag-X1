"""Voice-in extractive-answer CLI: an audio file path is transcribed via
Sarvam STT, then fed unchanged into the hot-path pipeline, then (only if
ANSWERED) reviewed by a non-blocking, informational Claude Sonnet 5
groundedness check. The reflect leg re-embeds and re-retrieves the top
chunk locally to get the source passage text, rather than touching
pipeline.py/pipeline_schemas.py directly.

Usage: uv run python -m scripts.voice_answer <audio_file> [language] [strategy]
"""

import sys
import time
from pathlib import Path

from qdrant_client import QdrantClient

from speculative_rag.config import BCP47_LANGUAGE_CODES, SETTINGS
from speculative_rag.embedding import Embedder
from speculative_rag.pipeline import run_pipeline
from speculative_rag.pipeline_schemas import RagOutcome
from speculative_rag.reflect import check_groundedness_llm
from speculative_rag.retrieval import Retriever
from speculative_rag.stt import transcribe_audio


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: uv run python -m scripts.voice_answer <audio_file> [language] [strategy]")
        return

    audio_path = Path(sys.argv[1])
    if not audio_path.is_file():
        print(f"No such audio file: {audio_path}")
        return

    language = sys.argv[2] if len(sys.argv) > 2 else SETTINGS.default_language
    strategy = sys.argv[3] if len(sys.argv) > 3 else SETTINGS.default_strategy
    collection_name = f"{language}_{strategy}"
    language_code = BCP47_LANGUAGE_CODES.get(language, SETTINGS.sarvam_default_language_code)

    print(f"Transcribing '{audio_path}' (language_code={language_code})...")
    stt_start = time.perf_counter()
    transcript = transcribe_audio(audio_path, language_code)
    stt_ms = (time.perf_counter() - stt_start) * 1000.0
    print(f"transcript: {transcript}")
    print(f"stt latency (ms): {stt_ms:.2f}\n")

    print(f"Loading embedder and connecting to collection '{collection_name}'...")
    embedder = Embedder()
    client = QdrantClient(path=str(SETTINGS.qdrant_path))
    if not client.collection_exists(collection_name):
        available = [c.name for c in client.get_collections().collections]
        print(f"No such collection '{collection_name}'. Available: {available}")
        print("Run `uv run python -m scripts.run_recall_eval` first to build the index.")
        return
    retriever = Retriever(client, collection_name)

    result = run_pipeline(
        transcript,
        embedder,
        retriever,
        off_topic_threshold=SETTINGS.off_topic_threshold,
        groundedness_threshold=SETTINGS.groundedness_threshold,
        top_k=SETTINGS.answer_top_k,
    )

    print(f"outcome: {result.outcome.value}")
    if result.outcome == RagOutcome.ANSWERED:
        print(f"answer: {result.answer.text}")
        print(
            f"source passage: {result.answer.source_passage_id}  "
            f"score={result.answer.retrieval_score:.3f}  "
            f"groundedness={result.answer.groundedness_score:.3f}"
        )
    else:
        print(f"refused: {result.refusal_reason}")

    print("latency (ms):")
    for timing in result.timings:
        print(f"  {timing.stage:24s} {timing.elapsed_ms:7.2f}")
    print(f"  {'TOTAL':24s} {result.total_ms:7.2f}\n")

    if result.outcome == RagOutcome.ANSWERED:
        print("--- review (informational, non-blocking; answer above is already final) ---")
        query_vector = embedder.embed_query(transcript)
        retrieval = retriever.search(query_vector, k=1)
        source_text = retrieval.top.chunk.text if retrieval.top is not None else ""

        reflect_start = time.perf_counter()
        verdict = check_groundedness_llm(result.answer.text, source_text)
        reflect_ms = (time.perf_counter() - reflect_start) * 1000.0
        print(f"grounded: {verdict.grounded}")
        print(f"reasoning: {verdict.reasoning}")
        print(f"async groundedness check latency (ms): {reflect_ms:.2f}")


if __name__ == "__main__":
    main()

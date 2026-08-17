import hashlib
from pathlib import Path

import pandas as pd

from speculative_rag.schemas import Passage, QueryRecord


def passage_id_for(language: str, text: str) -> str:
    digest = hashlib.sha1(f"{language}:{text}".encode("utf-8")).hexdigest()
    return digest[:16]


def _has_gold_passage(passages: dict) -> bool:
    return sum(int(x) for x in passages["is_selected"]) > 0


def load_validation_sample(parquet_path: Path, sample_size: int, seed: int) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path)
    df = df[df["passages"].apply(_has_gold_passage)]
    n = min(sample_size, len(df))
    return df.sample(n=n, random_state=seed).reset_index(drop=True)


def build_query_records(
    df: pd.DataFrame, language: str, text_field: str, query_field: str = "query"
) -> list[QueryRecord]:
    records = []
    for _, row in df.iterrows():
        passages = row["passages"]
        texts = list(passages[text_field])
        selected = [int(x) for x in passages["is_selected"]]
        passage_ids = [passage_id_for(language, t) for t in texts]
        relevant = [pid for pid, sel in zip(passage_ids, selected) if sel == 1]
        records.append(
            QueryRecord(
                query_id=int(row["query_id"]),
                language=language,
                query_type=row["query_type"],
                query_text=row[query_field],
                passage_ids=passage_ids,
                relevant_passage_ids=relevant,
            )
        )
    return records


def dedupe_passages(df: pd.DataFrame, language: str, text_field: str) -> dict[str, Passage]:
    passages: dict[str, Passage] = {}
    for _, row in df.iterrows():
        for text in row["passages"][text_field]:
            pid = passage_id_for(language, text)
            if pid not in passages:
                passages[pid] = Passage(
                    passage_id=pid,
                    text=text,
                    language=language,
                    query_type=row["query_type"],
                )
    return passages

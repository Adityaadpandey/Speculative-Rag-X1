import numpy as np
import pandas as pd

from speculative_rag.data import (
    build_query_records,
    dedupe_passages,
    load_validation_sample,
    passage_id_for,
)


def _make_row(query_id, query_type, translated, english, is_selected):
    return {
        "query_id": query_id,
        "query_type": query_type,
        "query": f"query {query_id}",
        "passages": {
            "Translated_passages": np.array(translated, dtype=object),
            "English_passages": np.array(english, dtype=object),
            "is_selected": np.array(is_selected, dtype=np.int32),
        },
    }


def _make_df():
    rows = [
        _make_row(1, "DESCRIPTION", ["t1a", "t1b"], ["e1a", "e1b"], [0, 1]),
        _make_row(2, "NUMERIC", ["t2a"], ["e2a"], [0]),  # no gold passage
        _make_row(3, "ENTITY", ["t1a", "t3b"], ["e1a", "e3b"], [1, 0]),  # t1a repeats row 1
    ]
    return pd.DataFrame(rows)


def test_passage_id_is_deterministic_and_language_scoped():
    a = passage_id_for("hi", "same text")
    b = passage_id_for("hi", "same text")
    c = passage_id_for("en", "same text")
    assert a == b
    assert a != c


def test_load_validation_sample_filters_zero_selected_rows(tmp_path):
    df = _make_df()
    path = tmp_path / "sample.parquet"
    df.to_parquet(path)

    sampled = load_validation_sample(path, sample_size=10, seed=42)

    assert set(sampled["query_id"]) == {1, 3}


def test_build_query_records_maps_gold_passages():
    df = _make_df()
    records = build_query_records(df, language="hi", text_field="Translated_passages")

    by_id = {r.query_id: r for r in records}
    assert len(records) == 3
    assert by_id[1].relevant_passage_ids == [passage_id_for("hi", "t1b")]
    assert by_id[2].relevant_passage_ids == []
    assert by_id[3].relevant_passage_ids == [passage_id_for("hi", "t1a")]


def test_dedupe_passages_collapses_repeated_text():
    df = _make_df()
    passages = dedupe_passages(df, language="hi", text_field="Translated_passages")

    # t1a appears in rows 1 and 3 but must collapse to a single passage
    assert len(passages) == 4
    assert passage_id_for("hi", "t1a") in passages

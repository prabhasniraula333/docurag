import pytest

from app.rag.retrieve import retrieve

RECORDS = [
    {"page": 1, "text": "A", "vector": [1.0, 0.0]},
    {"page": 2, "text": "B", "vector": [0.0, 1.0]},
    {"page": 3, "text": "C", "vector": [0.6, 0.8]},
    {"page": 4, "text": "D", "vector": [0.8, 0.6]},
]

QUESTION = [0.8, 0.6]


def test_returns_best_matches_first():
    results = retrieve(QUESTION, RECORDS, 2)

    assert [r["page"] for r in results] == [4, 3]
    assert results[0]["score"] == pytest.approx(1.0)
    assert results[1]["score"] == pytest.approx(0.96)


def test_k_more_than_chunks_returns_all_chunks_best_first():
    results = retrieve(QUESTION, RECORDS, 5)

    assert len(results) == 4 and [r["page"] for r in results] == [4, 3, 1, 2]


def test_result_only_holds_page_text_score_keys():
    results = retrieve(QUESTION, RECORDS, 2)

    assert set(results[0].keys()) == {"page", "text", "score"}


def test_k_of_0_and_negative_raises_value_error():

    with pytest.raises(ValueError):
        retrieve(QUESTION, RECORDS, 0)
    with pytest.raises(ValueError):
        retrieve(QUESTION, RECORDS, 0)


def test_mismatched_vector_lengths_raises_value_error():

    with pytest.raises(ValueError):
        retrieve([0.6, 0.8, 0.4], RECORDS, 2)


def test_an_empty_list_returns_an_empty_list():
    results = retrieve(QUESTION, [], 2)

    assert results == []

import pytest
from app.rag.chunk import chunk_text


def test_chunks_overlap_by_the_requested_amount():
    chunks = chunk_text("abcdefghijklmnopqrst", 8, 3)
    assert chunks == ["abcdefgh", "fghijklm", "klmnopqr", "pqrst"]


def test_overlap_not_smaller_than_size_raises_value_error():
    with pytest.raises(ValueError):
        chunk_text("hello", 5, 5)


def test_no_redundant_tail():
    chunks = chunk_text("abcdefghijklmnopqr", 8, 3)
    assert chunks[-1:] == ["klmnopqr"]


def test_short_text_stays_whole():
    chunks = chunk_text("hello", 8, 3)
    assert chunks == ["hello"]


def test_empty_text():
    chunks = chunk_text("", 8, 3)
    assert chunks == []


def test_zero_overlap_is_valid():
    chunks = chunk_text("abcdef", 3, 0)
    assert chunks == ["abc", "def"]

import pytest

from app.rag.embed import embed_texts


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def test_empty_list_returns_empty_list():
    vector = embed_texts([])
    assert vector == []


def test_bare_string_raises_error():
    with pytest.raises(TypeError):
        embed_texts("hello")


def test_vectors_have_length_one():
    vector = embed_texts(["the cat sat on the mat"])[0]
    length_squared = sum(x * x for x in vector)
    assert length_squared == pytest.approx(1.0, abs=1e-5)


def test_one_384_dimensions_vector_per_text():
    vectors = embed_texts(
        [
            "the cat sat on the mat",
            "the stock market crashed today",
            "the kitten is sitting on the wall",
        ]
    )
    assert len(vectors) == 3 and all(len(v) == 384 for v in vectors)


def test_same_text_gives_same_vector():
    first = embed_texts(["a football is round"])[0]
    second = embed_texts(["a football is round"])[0]
    assert first == pytest.approx(second)


def test_meaningful_texts_have_higher_score_than_unrelated():
    meaningful = dot(
        *embed_texts(["the cat sat on the mat", "the kitten sat on the rug"])
    )
    unrelated = dot(
        *embed_texts(["the cat sat on the mat", "the stock prices went up today"])
    )
    assert meaningful > unrelated + 0.2

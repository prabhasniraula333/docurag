import pytest
from pypdf import PdfWriter

from app.rag.embed import embed_texts
from app.rag.pipeline import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_OVERLAP,
    build_records,
    chunk_text,
    search,
)


def test_default_chunk_settings_are_valid():
    chunks = chunk_text("word " * 1000, DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP)
    assert len(chunks) > 1


def make_blank_pdf(path, num_pages):
    writer = PdfWriter()
    for _ in range(num_pages):
        writer.add_blank_page(width=200, height=200)
    with open(path, "wb") as f:
        writer.write(f)


def test_blank_question_is_rejected():
    with pytest.raises(ValueError):
        search("", [])


def test_empty_pdf_gives_no_records(tmp_path):
    pdf_path = tmp_path / "blank.pdf"
    make_blank_pdf(pdf_path, 1)
    assert build_records(str(pdf_path)) == []


def test_search_finds_the_matching_chunk():
    records = [
        {
            "page": 1,
            "text": "the cat sat on the mat",
            "vector": embed_texts(["the cat sat on the mat"])[0],
        },
        {
            "page": 2,
            "text": "stock prices fell sharply today",
            "vector": embed_texts(["stock prices fell sharply today"])[0],
        },
        {
            "page": 3,
            "text": "the football match ended in a draw",
            "vector": embed_texts(["the football match ended in a draw"])[0],
        },
    ]
    result = search("a kitten resting on a rug", records, 1)
    assert result[0]["text"] == "the cat sat on the mat"

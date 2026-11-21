from app.rag.pipeline import search,build_records
from app.rag.embed import embed_texts
import pytest
from tests.test_extract import make_blank_pdf

def test_blank_question_is_rejected():
    with pytest.raises(ValueError):
        search("",[])

def test_empty_pdf_gives_no_records(tmp_path):
    pdf_path= tmp_path / "blank.pdf"
    make_blank_pdf(pdf_path,1)
    assert build_records(str(pdf_path)) == []

def test_search_finds_the_matching_chunk():
    records=[{'page': 1,'text': 'the cat sat on the mat','vector': embed_texts(['the cat sat on the mat'])[0]},{'page': 2,'text': 'stock prices fell sharply today','vector': embed_texts(['stock prices fell sharply today'])[0]},{'page': 3,'text': 'the football match ended in a draw','vector': embed_texts(['the football match ended in a draw'])[0]}]
    result=search("a kitten resting on a rug", records, 1)
    assert result[0]["text"] == "the cat sat on the mat" 



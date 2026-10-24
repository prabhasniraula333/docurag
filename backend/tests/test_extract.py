from pypdf import PdfWriter

from app.rag.extract import extract_pages


def make_blank_pdf(path, num_pages):
    writer = PdfWriter()
    for _ in range(num_pages):
        writer.add_blank_page(width=200, height=200)
    with open(path, "wb") as f:
        writer.write(f)


def test_page_number_starts_at_one(tmp_path):
    pdf_path = tmp_path / "blank.pdf"
    make_blank_pdf(pdf_path, 3)

    pages = extract_pages(str(pdf_path))
    assert [number for number, _ in pages] == [1, 2, 3]


def test_blank_pages_are_kept_as_empty_text(tmp_path):
    pdf_path = tmp_path / "blank.pdf"
    make_blank_pdf(pdf_path, 2)

    pages = extract_pages(str(pdf_path))
    assert [text for _, text in pages] == ["", ""]

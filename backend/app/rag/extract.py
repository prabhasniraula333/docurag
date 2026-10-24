from pypdf import PdfReader


def extract_pages(path: str) -> list[tuple[int, str]]:
    reader = PdfReader(path)
    texts = []
    for i, page in enumerate(reader.pages, start=1):
        texts.append((i, page.extract_text()))
    return texts

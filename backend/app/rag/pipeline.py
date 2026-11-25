from app.rag.chunk import chunk_text
from app.rag.embed import embed_texts
from app.rag.extract import extract_pages
from app.rag.retrieve import retrieve

DEFAULT_CHUNK_SIZE = 800
DEFAULT_OVERLAP = 150


def build_records(
    path: str, size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_OVERLAP
) -> list[dict]:
    """Read a PDF and return one record per chunk: page, text and vector."""
    records = []
    for page_number, page_text in extract_pages(path):
        chunks = chunk_text(page_text, size, overlap)
        vectors = embed_texts(chunks)
        for chunk, vector in zip(chunks, vectors):
            records.append({"page": page_number, "text": chunk, "vector": vector})

    return records


def search(question: str, records: list[dict], k: int = 3) -> list[dict]:
    """Return the k chunks most similar to the question, best first."""
    if not question.strip():
        raise ValueError(
            "the question cannot be empty or blank.please enter a relevant question."
        )
    question_vector = embed_texts([question])[0]
    return retrieve(question_vector, records, k)

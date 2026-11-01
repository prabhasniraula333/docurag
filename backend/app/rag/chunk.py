def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """Split text into pieces of at most `size` characters.

    Each piece starts with the last `overlap` characters of the previous one.
    Raises ValueError unless size > 0 and 0 <= overlap < size.
    """
    if size == 0 or size < 0 or overlap >= size or overlap < 0:
        raise ValueError(
            f"invalid value: (size must not be zero or negative) size:{size},(overlap must be less than size and greater than zero) overlap:{overlap} "
        )
    step = size - overlap
    chunks = []
    for start in range(0, len(text), step):
        chunks.append(text[start : start + size])
        if start + size >= len(text):
            break
    return chunks

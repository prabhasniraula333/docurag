def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def retrieve(query_vector: list[float], records: list[dict], k: int) -> list[dict]:
    """Return the k records most similar to the query, best first.

    query_vector: the embedding of the question (for our model, 384 numbers).
    records: dictionaries like {"page": 3, "text": "...", "vector": [384 numbers]}.
    k: how many results to return.
    Each result is a dictionary: {"page": ..., "text": ..., "score": ...}.
    """
    if k <= 0:
        raise ValueError(
            f"something must be retireved.k must atleast be 1.it is currently{k}."
        )
    results = []
    for record in records:
        if len(query_vector) != len(record["vector"]):
            raise ValueError(
                f"two vectors differ in length.(query_length={len(query_vector)},record_length={len(record['vector'])})"
            )
        score = _dot(query_vector, record["vector"])
        results.append({"page": record["page"], "text": record["text"], "score": score})

    results = sorted(results, key=lambda p: p["score"], reverse=True)
    return results[:k]

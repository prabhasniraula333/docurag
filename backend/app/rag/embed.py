MODEL_NAME = "all-MiniLM-L6-v2"
_model = None


def _get_model():
    """Load the embedding model on first use and reuse it afterwards."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Return one vector (a list of floats) for each text, in the same order."""
    if isinstance(texts, str):
        raise TypeError("need a list of strings")
    if not texts:
        return []
    model = _get_model()
    return model.encode(texts, show_progress_bar=False).tolist()

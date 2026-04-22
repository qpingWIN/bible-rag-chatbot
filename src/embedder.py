"""Sentence-transformer embeddings using all-MiniLM-L6-v2."""

from sentence_transformers import SentenceTransformer
import numpy as np
import torch

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def _best_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_model() -> SentenceTransformer:
    device = _best_device()
    print(f"Loading embedding model: {MODEL_NAME}  (device: {device})")
    model = SentenceTransformer(MODEL_NAME, device=device)
    print(f"  Embedding dimension: {model.get_embedding_dimension()}")
    return model


def embed_texts(
    model: SentenceTransformer,
    texts: list[str],
    batch_size: int = 512,
    show_progress: bool = True,
) -> np.ndarray:
    """Return L2-normalised float32 embeddings, shape (N, dim)."""
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embeddings.astype(np.float32)

"""Sentence-transformer embeddings using BGE for retrieval.

BGE models are trained for asymmetric retrieval: queries get a
short instruction prefix, documents don't. The two helpers
`embed_documents` and `embed_query` enforce this at the API
level so callers don't have to remember.

Reference: https://huggingface.co/BAAI/bge-base-en-v1.5
"""

from sentence_transformers import SentenceTransformer
import numpy as np
import torch

MODEL_NAME = "BAAI/bge-base-en-v1.5"

# BGE's recommended query instruction prefix, prepended to query
# Document text passes through unmodified.
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


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
    print(f"  Embedding dimension: {model.get_sentence_embedding_dimension()}")
    return model


def embed_texts(
    model: SentenceTransformer,
    texts: list[str],
    batch_size: int = 64,
    show_progress: bool = True,
) -> np.ndarray:
    """Return L2-normalised float32 embeddings, shape (N, dim).
    
    For document embeddings, call this directly (no prefix). For
    queries, use embed_query() which adds the BGE instruction prefix.
    """
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embeddings.astype(np.float32)


def embed_query(model: SentenceTransformer, query: str) -> np.ndarray:
   #Embed a single query string with BGE's instruction prefix
   #Returns shape (1, dim) 
    prefixed = QUERY_INSTRUCTION + query
    return embed_texts(model, [prefixed], show_progress=False)
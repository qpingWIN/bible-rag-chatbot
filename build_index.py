"""
One-time script: loading all data, then chunking, followed by embedding and saving FAISS index.

Run this before using the chatbot: python build_index.py

Takes ~2-5 minutes on first run (downloads embedding model, processes ~65k chunks).
Subsequent runs load from cache in <5 seconds.

WHAT THIS SCRIPT DOES (step by step)
--------------------------------------
1. Load raw data: KJV verses, BSB verses, MHC commentary into Document objects
2. Chunk: bible verses are already atomic (1 verse = 1 chunk). MHC commentary is split into overlapping 250-word windows (see src/chunker.py)
3. Embed: run every chunk through all-MiniLM-L6-v2 to get the float32 matrix (N, 384)
4. Build FAISS index: store the matrix in an IndexFlatIP for fast dot-product search
5. Build BM25 index using rank_bm25
5. Save: write indexes + chunk metadata to data/index/ so the app loads instantly
"""

import sys
from pathlib import Path

# make "src" importable
sys.path.insert(0, str(Path(__file__).parent))

from src.ingest import load_all
from src.chunker import build_all_chunks
from src.embedder import load_model, embed_texts
from src.retriever import build_index, save_index, index_exists
from src.bm25_retriever import (
    build_bm25_index,
    save_bm25_index,
    bm25_index_exists,
)

def main():
    if index_exists() and bm25_index_exists():
        print("Both indexes already exist at data/index/. Delete to rebuild.")
        print("  rm -rf data/index/")
        return

    # Loading
    kjv_docs, bsb_docs, mhc_docs, xrefs = load_all()

    # Chunking
    chunks = build_all_chunks(kjv_docs, bsb_docs, mhc_docs)
    #Build FAISS if required
    if not index_exists():
        # Embedding
        model = load_model() 
        texts = [c.text for c in chunks]
        print(f"\nEmbedding {len(texts):,} chunks (this takes a few minutes)...")
        embeddings = embed_texts(model, texts)
        print(f"Embeddings shape: {embeddings.shape}") 
        index = build_index(chunks, embeddings)
        save_index(index, chunks)
    #Build bm25 if required    
    if not bm25_index_exists():
        chunks_dict = [c.to_dict() for c in chunks]
        bm25 = build_bm25_index(chunks_dict)
        save_bm25_index(bm25)

    print("\nDone. Run the chatbot with:")
    print("  python app.py")


if __name__ == "__main__":
    main()

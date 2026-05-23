"""BM25 lexical retrieval over the same chunks as the FAISS index

BM25 is a keyword-based scoring function that complements dense embedding
retrieval. It's good at exact name matches, short documents(verses are really short in our case)
and exact keyword overlap (exactly where dense embeddings tend to be weakest)

The index is built once over all chunk texts and saved alongside
the FAISS index. Load it at runner startup just like the FAISS index.
"""


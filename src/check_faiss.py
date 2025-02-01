'''
Check the faiss database of embeddings from the json data (original data as pdfs)
'''

import faiss
import numpy as np

def load_faiss_index(index_path="../data/vector_index.faiss"):
    """Load the FAISS index and check its structure."""
    index = faiss.read_index(index_path)
    
    # Get the number of vectors (documents) stored
    num_vectors = index.ntotal

    # Get the embedding dimension (e.g., 384 for MiniLM)
    embedding_dim = index.d

    print(f"✅ FAISS index contains {num_vectors} documents.")
    print(f"✅ Embedding dimension: {embedding_dim}")

    return index

if __name__ == "__main__":
    index = load_faiss_index()

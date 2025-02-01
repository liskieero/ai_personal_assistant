'''
Create embeddings from the saved json file (own data)
and store them into a vector database (FAISS)
'''

import faiss
import numpy as np
from embeddings import get_embedding
from load_text import load_extracted_text

def store_embeddings(json_path, index_path="../data/vector_index.faiss"):
    """Convert document text into embeddings and store them in FAISS."""
    data = load_extracted_text(json_path)
    embeddings = []
    doc_ids = list(data.keys())

    for text in data.values():
        embeddings.append(get_embedding(text))

    # Convert list of embeddings to NumPy array
    embeddings_np = np.array(embeddings).astype("float32")

    # Create a FAISS index
    index = faiss.IndexFlatL2(embeddings_np.shape[1])
    index.add(embeddings_np)

    # Save FAISS index
    faiss.write_index(index, index_path)
    print(f"✅ Stored {len(embeddings)} document embeddings in FAISS.")

if __name__ == "__main__":
    store_embeddings("../data/extracted_text.json")

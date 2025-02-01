from sentence_transformers import SentenceTransformer
import numpy as np

# Load the embedding model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def get_embedding(text):
    """Generate an embedding for a given text."""
    return model.encode(text, convert_to_numpy=True)

if __name__ == "__main__":
    test_text = "This fund invests in technology companies."
    embedding = get_embedding(test_text)
    print(f"Embedding shape: {embedding.shape}")  # Should print (384,)
    print(f"First 5 values: {embedding[:5]}")

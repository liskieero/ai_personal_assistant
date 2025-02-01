'''
User queries the faiss, best document gets returned (based on embeddings),
then closest sentence within that document gets returned.
'''

import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer

# Load embedding model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def load_faiss_index(index_path="../data/vector_index.faiss"):
    """Load FAISS index."""
    return faiss.read_index(index_path)

def load_extracted_text(json_path="../data/extracted_text.json"):
    """Load extracted document texts from JSON."""
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def find_closest_document(query, index, doc_texts):
    """Find the closest document in FAISS based on query embedding."""
    query_embedding = model.encode(query).astype("float32").reshape(1, -1)
    
    # Search FAISS index for closest document
    distances, closest_doc_idx = index.search(query_embedding, 1)  # Get closest match index

    # Get document name and text
    doc_name = list(doc_texts.keys())[closest_doc_idx[0][0]]
    doc_text = doc_texts[doc_name]
    
    return doc_name, doc_text, distances[0][0]  # Also return similarity score

def find_best_matching_sentence(query, document_text):
    """Find the most relevant sentence in the document using embeddings."""
    sentences = document_text.split(".")  # Split document into sentences
    sentence_embeddings = np.array([model.encode(s) for s in sentences])  # Embed sentences

    query_embedding = model.encode(query)  # Embed query
    similarities = np.dot(sentence_embeddings, query_embedding)  # Compute similarity scores

    best_match_index = np.argmax(similarities)  # Find the most similar sentence
    return sentences[best_match_index] if sentences else "No relevant answer found."

if __name__ == "__main__":
    # Load FAISS index and document texts
    index = load_faiss_index()
    doc_texts = load_extracted_text()

    # Ask user for a query
    query = input("\n🔍 Enter your question: ")

    # Find best matching document
    doc_name, doc_text, score = find_closest_document(query, index, doc_texts)
    print(f"\n✅ Closest document: {doc_name} (Similarity Score: {score:.4f})")

    # Find best matching sentence
    best_sentence = find_best_matching_sentence(query, doc_text)
    print(f"\n✅ Answer: {best_sentence}")

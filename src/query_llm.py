'''
User inputs a query
query gets embedded and compared to embedded saved documents
most similar document gets retrieved
best document gets chunked with overlap and passed to an llm
the llm response with highest confidence score gets returned
'''

import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# Load embedding model and LLM pipeline
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

# Define chunking parameters
CHUNK_SIZE = 256  # Max tokens per chunk
OVERLAP = 128  # Overlap between chunks to avoid cutting answers

def load_faiss_index(index_path="../data/vector_index.faiss"):
    """Load FAISS index."""
    return faiss.read_index(index_path)

def load_extracted_text(json_path="../data/extracted_text.json"):
    """Load extracted document texts from JSON."""
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def find_closest_document(query, index, doc_texts):
    """Find the closest document in FAISS based on query embedding."""
    query_embedding = embedding_model.encode(query).astype("float32").reshape(1, -1)
    
    # Search FAISS index for closest document
    distances, closest_doc_idx = index.search(query_embedding, 1)  # Get closest match index

    # Get document name and text
    doc_name = list(doc_texts.keys())[closest_doc_idx[0][0]]
    doc_text = doc_texts[doc_name]
    
    return doc_name, doc_text, distances[0][0]  # Also return similarity score

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    """Break text into overlapping chunks."""
    words = text.split()  # Tokenize based on spaces (simplified)
    chunks = []
    i = 0
    
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        i += (chunk_size - overlap)  # Move forward by chunk size minus overlap
    
    return chunks

def find_best_answer(query, document_text):
    """Find the most relevant answer using an LLM across document chunks."""
    chunks = chunk_text(document_text)  # Break document into chunks
    
    best_answer = None
    best_score = 0
    
    for chunk in chunks:
        try:
            response = qa_pipeline(question=query, context=chunk)
            answer = response["answer"]
            score = response["score"]

            if score > best_score:  # Keep the highest confidence answer
                best_answer = answer
                best_score = score

        except Exception as e:
            print(f"Error processing chunk: {e}")

    return best_answer if best_answer else "No relevant answer found."

if __name__ == "__main__":
    # Load FAISS index and document texts
    index = load_faiss_index()
    doc_texts = load_extracted_text()

    # Ask user for a query
    query = input("\n🔍 Enter your question: ")

    # Find best matching document
    doc_name, doc_text, score = find_closest_document(query, index, doc_texts)
    print(f"\n✅ Closest document: {doc_name} (Similarity Score: {score:.4f})")

    # Find best answer from chunks
    best_answer = find_best_answer(query, doc_text)
    print(f"\n✅ Answer: {best_answer}")

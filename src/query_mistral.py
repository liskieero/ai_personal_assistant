'''
User inputs a query
query gets embedded and compared to embedded saved documents
most similar document gets retrieved
best document gets chunked with overlap and passed to a GenAI LLM
the GenAI LLM response with highest confidence score gets returned
'''

import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load embedding model
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Load Mistral-7B model and tokenizer
model_name = "mistralai/Mistral-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

# Define chunking parameters
CHUNK_SIZE = 512  # Larger chunk for more context
OVERLAP = 128  # Keeps some overlap to avoid missing partial answers

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
    distances, closest_doc_idx = index.search(query_embedding, 1)

    # Get document name and text
    doc_name = list(doc_texts.keys())[closest_doc_idx[0][0]]
    doc_text = doc_texts[doc_name]
    
    return doc_name, doc_text, distances[0][0]

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    """Break text into overlapping chunks."""
    words = text.split()
    chunks = []
    i = 0
    
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        i += (chunk_size - overlap)
    
    return chunks

def generate_answer(query, document_text):
    """Generate an answer using Mistral-7B across document chunks."""
    chunks = chunk_text(document_text)

    best_answer = None
    best_score = 0

    for chunk in chunks:
        input_text = f"Context: {chunk}\n\nQuestion: {query}\n\nAnswer:"
        inputs = tokenizer(input_text, return_tensors="pt")

        # Generate response
        output = model.generate(**inputs, max_new_tokens=100)
        answer = tokenizer.decode(output[0], skip_special_tokens=True)

        # Simple length-based confidence scoring (longer responses are often better)
        score = len(answer)

        if score > best_score:
            best_answer = answer
            best_score = score

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

    # Generate answer from Mistral
    best_answer = generate_answer(query, doc_text)
    print(f"\n✅ Answer: {best_answer}")

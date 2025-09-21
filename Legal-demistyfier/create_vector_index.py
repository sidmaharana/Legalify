# create_vector_index.py

import os
import json
import time
import pickle
from langchain_google_vertexai import VertexAIEmbeddings

# Configuration
CHUNKS_FILE = "knowledge_base_chunks.json"
INDEX_FILE = "vector_index.pkl"
EMBEDDING_MODEL = "text-embedding-004"


def main():
    """
    Reads text chunks, generates embeddings using the Gemini API,
    and saves them to a searchable index file.
    """
    print("🚀 Starting vector index creation...")

    # Load the chunked text data
    if not os.path.exists(CHUNKS_FILE):
        print(f"❌ Error: Chunks file '{CHUNKS_FILE}' not found.")
        print("Please run 'python chunk_legal_text.py' first.")
        return
    with open(CHUNKS_FILE, 'r', encoding='utf-8') as f:
        chunks = json.load(f)

    # Initialize Vertex AI Embeddings model
    embeddings_model = VertexAIEmbeddings(model_name=EMBEDDING_MODEL)

    print(f"Found {len(chunks)} text chunks to embed.")
    indexed_data = []

    for title, text in chunks.items():
        print(f"  - Embedding chunk: '{title}'...")
        try:
            # Call Vertex AI Embeddings to get the embedding vector for the text
            embedding = embeddings_model.embed_query(text)

            # Store the original title, text, and its new embedding
            indexed_data.append({
                "title": title,
                "text": text,
                "embedding": embedding
            })
            time.sleep(2) # Increased delay to 2 seconds

        except Exception as e:
            print(f"⚠️ Could not process chunk '{title}'. Error: {e}")

    # Save the entire index to a file using pickle
    with open(INDEX_FILE, 'wb') as f:
        pickle.dump(indexed_data, f)

    print(
        f"\n✅ Success! Created vector index with {len(indexed_data)} entries.")
    print(f"Output saved to '{INDEX_FILE}'.")


if __name__ == "__main__":
    main()

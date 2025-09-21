# analyzer.py

import os
import json
import numpy as np
from langchain_google_vertexai import VertexAIEmbeddings, VertexAI


# This is the core RAG function. It takes a query and the indexed data,
# and returns the most relevant text chunks.
def find_relevant_chunks(query: str, indexed_data: list, embeddings_model: VertexAIEmbeddings, top_k: int = 3):
    """Finds the top_k most relevant text chunks from the index for a given query."""

    # 1. Embed the search query using VertexAIEmbeddings
    query_embedding = embeddings_model.embed_query(query)

    # 2. Calculate similarity (dot product) between the query and all chunks
    dot_products = np.dot(
        np.stack([item['embedding'] for item in indexed_data]),
        query_embedding)

    # 3. Get the indices of the top_k most similar chunks
    top_k_indices = np.argsort(dot_products)[-top_k:][::-1]

    # 4. Return the top_k chunks themselves
    return [indexed_data[i] for i in top_k_indices]


# The prompt is updated to work with the retrieved context.
RAG_PROMPT_TEMPLATE = """
You are a legal analyst AI. Your task is to explain a clause from a user's document.
Use the provided "RETRIEVED KNOWLEDGE" from a trusted knowledge base to provide a comprehensive explanation.

**Instructions:**
1.  Analyze the "USER'S CLAUSE".
2.  Review the "RETRIEVED KNOWLEDGE" which contains relevant sections from standard legal documents (like the MIT or Apache licenses).
3.  Synthesize the information to provide a clear explanation and analysis of the user's clause.
4.  Structure your output as a single JSON object with the keys "explanation" and "impact".

---
**RETRIEVED KNOWLEDGE:**
{retrieved_knowledge}
---
**USER'S CLAUSE:**
{users_clause}
---
**YOUR JSON OUTPUT:**
"""


def get_rag_analysis(users_clause: str,
                     retrieved_knowledge: str, llm: VertexAI) -> dict | str:
    """Generates an explanation for a clause using the RAG pipeline."""
    try:
        prompt = RAG_PROMPT_TEMPLATE.format(
            retrieved_knowledge=retrieved_knowledge, users_clause=users_clause)

        response = llm.invoke(prompt)
        # We expect a single JSON object here, not a list.
        cleaned_response = response.strip().replace("```json",
                                                         "").replace(
                                                             "```", "")
        return json.loads(cleaned_response)

    except json.JSONDecodeError:
        return f"ERROR: Failed to parse JSON from the model's response. Response was:\n{response}"
    except Exception as e:
        return f"An error occurred during analysis: {e}"


# In analyzer.py, add this new function. The other functions remain the same.


def discover_concepts_with_ai(text_to_analyze: str, llm: VertexAI) -> list | str:
    """
    Uses an LLM to read a document and identify the key legal concepts within it.
    """
    discovery_prompt = f"""
    You are a legal analyst. Read the following legal text and identify the core, distinct legal concepts it contains.
    Return your findings as a simple, valid JSON list of strings. Each string should be a concise summary of a concept.

    Example:
    - Input Text: "The software is provided AS-IS. You may copy and modify it. All patent rights are granted."
    - Your JSON Output: ["Disclaimer of Warranty", "Permissions to copy and modify", "Grant of Patent Rights"]

    ---
    LEGAL TEXT:
    {text_to_analyze}
    ---
    YOUR JSON LIST:
    """
    try:
        response = llm.invoke(discovery_prompt)
        cleaned_response = response.strip().replace("```json",
                                                         "").replace(
                                                             "```", "")
        return json.loads(cleaned_response)
    except Exception as e:
        return f"Error during concept discovery: {e}"

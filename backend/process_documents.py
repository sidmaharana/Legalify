# process_documents.py

import os
import json
import pickle
from analyzer import find_relevant_chunks, get_rag_analysis, discover_concepts_with_ai  # Import the new function

# ... (Configuration and other parts of the script remain the same) ...
INPUT_DIR = "input_documents"
OUTPUT_DIR = "output_reports"
INDEX_FILE = "vector_index.pkl"


def main():
    print("🚀 Starting AI-Powered RAG Analysis...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Load the searchable vector index
    if not os.path.exists(INDEX_FILE):
        print(f"❌ ERROR: Vector index '{INDEX_FILE}' not found.")
        return
    with open(INDEX_FILE, 'rb') as f:
        indexed_data = pickle.load(f)
    print(f"✅ Vector index loaded with {len(indexed_data)} entries.")

    files_to_process = [f for f in os.listdir(INPUT_DIR) if f.endswith(".txt")]
    if not files_to_process:
        print(f"🟡 Warning: No .txt files found in '{INPUT_DIR}'.")
        return

    for filename in files_to_process:
        input_filepath = os.path.join(INPUT_DIR, filename)
        output_filepath = os.path.join(
            OUTPUT_DIR, f"{os.path.splitext(filename)[0]}_rag_analysis.txt")

        print(f"\nProcessing '{filename}'...")
        with open(input_filepath, 'r', encoding='utf-8') as f:
            raw_text = f.read()

        # 2. **UPGRADED DISCOVERY STEP**
        # Use the AI to discover key concepts in the document.
        print("  - Discovering key concepts with AI...")
        search_queries = discover_concepts_with_ai(raw_text)

        if isinstance(search_queries, str):  # Handle discovery errors
            print(f"  - Could not discover concepts: {search_queries}")
            continue

        report_content = [f"### AI-Powered RAG Analysis for: {filename}\n"]
        report_content.append(
            f"**Discovered Concepts:** {', '.join(search_queries)}\n")

        for query in set(search_queries):
            print(f"  - Analyzing concept: '{query}'")

            # 3. Retrieve relevant knowledge from the index for each concept
            relevant_chunks = find_relevant_chunks(query, indexed_data)
            retrieved_knowledge = "\n\n".join(
                [chunk['text'] for chunk in relevant_chunks])

            # 4. Generate the final analysis using the retrieved knowledge
            analysis = get_rag_analysis(query, retrieved_knowledge)

            # 5. Format the report
            report_content.append(
                f"\n--- Analysis for Concept: \"{query}\" ---\n")
            if isinstance(analysis, dict):
                report_content.append(
                    f"**Explanation:** {analysis.get('explanation', 'N/A')}")
                report_content.append(
                    f"**Potential Impact:** {analysis.get('impact', 'N/A')}\n")
            else:
                report_content.append(f"**Error:** {analysis}\n")

        with open(output_filepath, 'w', encoding='utf-8') as f:
            f.write("\n".join(report_content))

        print(f"✅ Success! RAG report saved to '{output_filepath}'")

    print("\n🎉 All documents processed.")


if __name__ == "__main__":
    main()

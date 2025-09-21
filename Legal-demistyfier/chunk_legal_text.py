# chunk_legal_text.py

import os
import re
import json

# Configuration
SOURCE_DATA_DIR = "knowledge_base_source"
CHUNKED_DATA_FILE = "knowledge_base_chunks.json"


def chunk_text(text: str) -> dict:
    """
    Splits a legal text into a dictionary of chunks using a more robust method.
    It finds all headers and treats the text between them as the content.
    """
    chunks = {}

    # This improved regex finds common heading patterns:
    # 1. "Section" or "Article" followed by numbers/letters.
    # 2. A number and a period at the start of a line (like in the GPL).
    # 3. Specific keywords like "Preamble" or "APPENDIX".
    pattern = re.compile(
        r'(^\s*(?:Section|Article)\s+[0-9A-ZIVXLC]+\..*|^\s*\d+\.\s+.*|^\s*Preamble\s*$|^\s*APPENDIX:.*)',
        re.MULTILINE)

    # Find all header matches and their positions
    headers = list(pattern.finditer(text))

    if not headers:  # If no identifiable sections were found
        return {"Full Document": text.strip()}

    # Extract content between headers
    for i, match in enumerate(headers):
        # Clean up the title
        title = match.group(0).strip().replace('\n', ' ')

        # Determine the start and end of the content
        start_pos = match.end()
        end_pos = headers[i + 1].start() if i + 1 < len(headers) else len(text)

        content = text[start_pos:end_pos].strip()

        if title and content:
            chunks[title] = content

    return chunks


def main():
    """
    Reads all .txt files from the source directory, chunks them,
    and saves the combined result to a single JSON file with unique keys.
    """
    print(f"🚀 Starting to chunk documents from '{SOURCE_DATA_DIR}'...")

    if not os.path.isdir(SOURCE_DATA_DIR):
        print(
            f"❌ Error: Source directory '{SOURCE_DATA_DIR}' not found. Please create it."
        )
        return

    all_chunks = {}
    files_to_process = [
        f for f in os.listdir(SOURCE_DATA_DIR) if f.endswith(".txt")
    ]

    if not files_to_process:
        print(f"🟡 Warning: No .txt files found in '{SOURCE_DATA_DIR}'.")
        return

    for filename in files_to_process:
        filepath = os.path.join(SOURCE_DATA_DIR, filename)
        print(f"  - Chunking '{filename}'...")
        with open(filepath, 'r', encoding='utf-8') as f:
            raw_text = f.read()

        document_chunks = chunk_text(raw_text)

        # Create unique keys for each chunk to prevent overwriting
        for title, text in document_chunks.items():
            unique_key = f"{os.path.splitext(filename)[0]}_{title}"
            all_chunks[unique_key] = text

    with open(CHUNKED_DATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, indent=2)

    print(
        f"\n✅ Success! All documents chunked into {len(all_chunks)} sections.")
    print(f"Output saved to '{CHUNKED_DATA_FILE}'.")


if __name__ == "__main__":
    main()

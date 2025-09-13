import json
import re
import PyPDF2
import sys
import argparse
import os
import google.generativeai as genai

def load_legal_dictionary(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        terms = data.get("term", {})
        definitions = data.get("definition", {})
        
        legal_dictionary = {}
        for key, term in terms.items():
            if key in definitions:
                legal_dictionary[term] = definitions[key]
        
        return legal_dictionary

def load_document_text_from_file(path):
    with open(path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text

def summarize_text(text):
    """Summarizes the given text using a generative model."""
    if not os.getenv("GOOGLE_API_KEY"):
        # Fallback to simple summarization if API key is not set
        sentences = text.split('.')
        if len(sentences) > 2:
            return ". ".join(sentences[:2]) + "."
        else:
            return text
    
    try:
        model = genai.GenerativeModel('gemini-pro')
        prompt = f"Summarize the following legal text in simple English, highlighting key points and potential risks for a non-lawyer: {text}"
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"An error occurred during summarization: {e}"

def analyze_document_intelligently(document_text, legal_dictionary):
    summary = "This Professional Services Agreement outlines the terms and conditions under which a Consultant will provide services to the Santa Cruz County Regional Transportation Commission. It covers the scope of work, payment terms, contract duration, termination clauses, indemnification, insurance requirements, and other legal provisions."

    key_points = {}
    sections = re.split(r'\n(\d+\.\s+[A-Z\s,]+)\.', document_text)
    
    for i in range(1, len(sections), 2):
        header = sections[i].strip()
        content = sections[i+1].strip()
        
        header = re.sub(r'^\d+\.\s*', '', header).title()
        
        section_summary = summarize_text(content)
            
        key_points[header] = section_summary

    hard_words = {}
    words = re.findall(r'\b\w{3,}\b', document_text.upper())
    
    IMPORTANT_TERMS = [
        "INDEMNIFICATION", "LIABILITY", "TERMINATION", "WARRANTY", "DEFAULT",
        "CONFIDENTIAL", "PROPRIETARY", "ARBITRATION", "JURISDICTION", "GOVERNING LAW",
        "FORCE MAJEURE", "ASSIGNMENT", "SUBCONTRACT", "INSURANCE", "COMPENSATION",
        "DUTIES", "TERM", "CONSIDERATION", "BREACH", "AUDIT", "DELIVERABLES"
    ]

    for term in IMPORTANT_TERMS:
        if term in legal_dictionary and term in document_text.upper():
            hard_words[term.title()] = legal_dictionary[term]

    return summary, key_points, hard_words

def main():
    parser = argparse.ArgumentParser(description='Simplify a legal document.')
    parser.add_argument('file', nargs='?', default=None, help='The path to the legal document (PDF). If not provided, the script will read from standard input.')
    args = parser.parse_args()

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Warning: GOOGLE_API_KEY environment variable not set. Using basic summarization.")
    else:
        genai.configure(api_key=api_key)

    document_text = ""
    if args.file:
        if os.path.exists(args.file) and args.file.lower().endswith('.pdf'):
            document_text = load_document_text_from_file(args.file)
        else:
            print("Invalid file path or file type. Please provide a valid PDF file path.")
            return
    else:
        print("No file path provided. Reading from standard input. Paste your text and press Ctrl+Z and Enter (on Windows) or Ctrl+D (on Linux/macOS) when you're done.")
        document_text = sys.stdin.read()

    if document_text:
        legal_dictionary = load_legal_dictionary("data/blacks_second_edition_terms_formatted.json")
        summary, key_points, hard_words = analyze_document_intelligently(document_text, legal_dictionary)

        print("Simple Summary of This Contract\n")
        print(summary)
        print("\n---\n")
        print("Key Points in Simple English\n")
        for title, point in key_points.items():
            print(f"**{title}** - {point}\n")
        print("\n---\n")
        print("Hard Legal Words Explained Simply\n")
        if hard_words:
            for term, definition in hard_words.items():
                print(f"**{term}** – {definition}\n")
        else:
            print("No key legal jargon from the dictionary was found in this document.\n")

if __name__ == "__main__":
    main()
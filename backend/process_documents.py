# process_documents.py

import os
import json
import pickle
import PyPDF2
from docx import Document
import re
from langchain_google_vertexai import VertexAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

def _extract_text_from_pdf(file_path):
    text = ""
    try:
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() or ""
    except Exception as e:
        print(f"Error extracting text from PDF {file_path}: {e}")
    return text

def _extract_text_from_docx(file_path):
    text = []
    try:
        doc = Document(file_path)
        for para in doc.paragraphs:
            text.append(para.text)
    except Exception as e:
        print(f"Error extracting text from DOCX {file_path}: {e}")
    return "\n".join(text)

def find_keywords_with_prerequisites(document_path, keywords, llm: VertexAI):
    found_data = {"keywords_found": {}}

    file_extension = os.path.splitext(document_path)[1].lower()
    document_text = ""

    if file_extension == ".pdf":
        document_text = _extract_text_from_pdf(document_path)
    elif file_extension == ".docx":
        document_text = _extract_text_from_docx(document_path)
    else:
        return {"error": f"Unsupported file type: {file_extension}"}

    if not document_text:
        return {"error": "Could not extract text from document."}

    # --- Vertex AI based Prerequisite Detection ---
    prompt_template = """
You are an expert legal assistant. Analyze the following document and identify the provided keywords. For each keyword found, determine if it has any prerequisites (other keywords from the provided list that must be understood or fulfilled before the current keyword). Also, provide a brief explanation of the relationship.

Document:
{document_text}

Keywords to identify and find prerequisites for: {keywords}

Provide the output as a JSON object with the following structure:
{{
  "analysis": [
    {{
      "keyword": "[Identified Keyword]",
      "sentence": "[Sentence where keyword was found]",
      "prerequisites": [
        {{
          "keyword": "[Prerequisite Keyword]",
          "explanation": "[Brief explanation of why it's a prerequisite]"
        }}
      ]
    }}
  ]
}}
If no prerequisites are found for a keyword, the "prerequisites" array should be empty.
"""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["document_text", "keywords"]
    )

    llm_chain = LLMChain(llm=llm, prompt=prompt)

    try:
        response = llm_chain.run(document_text=document_text, keywords=", ".join(keywords))
        # Attempt to parse the JSON response from the LLM
        ai_analysis = json.loads(response)
        return ai_analysis
    except Exception as e:
        print(f"Error during Vertex AI analysis: {e}")
        return {"error": f"Failed to get AI analysis: {e}"}

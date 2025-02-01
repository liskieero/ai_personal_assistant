'''
Reads alls the pdfs and converts them to a single json file
'''

import os
import fitz  # PyMuPDF
import json


def extract_text_from_pdfs(folder_path):
    """Extract text from all PDFs in a folder."""
    extracted_text = {}

    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(folder_path, filename)

            with fitz.open(pdf_path) as doc:
                text = "\n".join([page.get_text("text") for page in doc])
                extracted_text[filename] = text

    return extracted_text

def save_extracted_text_to_json(text_data, output_file):
    """Save extracted text as JSON."""
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(text_data, f, indent=4, ensure_ascii=False)

# Example usage
if __name__ == "__main__":
    folder = "../data"  # Adjust path as needed
    output_file = "../data/extracted_text.json"

    pdf_texts = extract_text_from_pdfs(folder)
    save_extracted_text_to_json(pdf_texts, output_file)

    print(f"\n✅ Extracted text saved to {output_file}")
     
    # for pdf, text in pdf_texts.items():
    #     print(f"\nExtracted from {pdf}:\n{text[:500]}...")  # Print first 500 characters

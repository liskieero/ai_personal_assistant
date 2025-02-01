'''
Loads text to a dictionary
'''

import json

def load_extracted_text(json_path):
    """Load extracted text from JSON file."""
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

if __name__ == "__main__":
    data = load_extracted_text("../data/extracted_text.json")
    print(f"Loaded {len(data)} documents.")

"""
src/ingest.py
Handles loading and extracting text from PDF documents using PyMuPDF.
Each page is preserved with its page number metadata.
"""

import os
import re
import pymupdf


def clean_text(raw_text: str) -> str:
    """
    Cleans raw extracted text from PDF artifacts while preserving
    legal article numbers, clauses, and structure.
    """
    if not raw_text:
        return ""

    # Replace unicode replacement characters (like \ufffd) often found in older PDF font encodings
    cleaned = raw_text.replace("\ufffd", "-")

    # Replace non-breaking spaces or tabs with regular space
    cleaned = cleaned.replace("\xa0", " ").replace("\t", " ")

    # Collapse sequences of 3 or more newlines into double newlines (keeps paragraph separation)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    # Strip excessive leading and trailing whitespace
    return cleaned.strip()


def extract_pages_from_pdf(pdf_path: str) -> list[dict]:
    """
    Opens a PDF and extracts text page by page.
    Returns a list of dictionaries with page text and metadata.

    Output format:
    [
        {
            "page_number": 1,
            "text": "...",
            "source": "constitution.pdf"
        },
        ...
    ]
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at: {pdf_path}")

    doc = pymupdf.open(pdf_path)
    total_pages = len(doc)
    extracted_pages = []

    filename = os.path.basename(pdf_path)

    for page_index in range(total_pages):
        page = doc[page_index]
        page_number = page_index + 1  # 1-indexed for human readability

        # Extract text from this page
        raw_text = page.get_text("text")
        cleaned_text = clean_text(raw_text)

        # Only retain pages that contain actual text content
        if cleaned_text:
            extracted_pages.append({
                "page_number": page_number,
                "text": cleaned_text,
                "source": filename
            })

    doc.close()
    return extracted_pages


if __name__ == "__main__":
    pdf_file = os.path.join("data", "constitution.pdf")
    print(f"Opening and extracting: {pdf_file} ...")

    pages = extract_pages_from_pdf(pdf_file)
    print(f"Successfully extracted {len(pages)} pages with text.\n")

    # Let's inspect Page 32 (The Preamble)
    preamble_page = next((p for p in pages if p["page_number"] == 32), None)
    if preamble_page:
        print("=" * 60)
        print(f"--- SAMPLE INSPECTION: PAGE {preamble_page['page_number']} (PREAMBLE) ---")
        print("=" * 60)
        print(preamble_page["text"][:500])
        print("...\n")

    # Let's inspect Page 42 (Article 21: Protection of Life & Liberty)
    article_21_page = next((p for p in pages if p["page_number"] == 42), None)
    if article_21_page:
        print("=" * 60)
        print(f"--- SAMPLE INSPECTION: PAGE {article_21_page['page_number']} (ARTICLE 21) ---")
        print("=" * 60)
        # Find where Article 21 begins on this page
        text = article_21_page["text"]
        idx = text.find("21. Protection of life")
        if idx != -1:
            print(text[idx:idx + 400])
        else:
            print(text[:400])
        print("...\n")

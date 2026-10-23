"""
src/chunking.py
Splits extracted document pages into smaller, semantically coherent chunks
with configurable chunk size and overlap, while preserving page metadata.
"""

import os
from typing import List, Dict


def split_text_with_overlap(
    text: str,
    chunk_size: int = 800,
    chunk_overlap: int = 150
) -> List[str]:
    """
    Splits text into chunks of maximum `chunk_size` characters,
    with `chunk_overlap` character overlap between consecutive chunks.
    
    Prioritizes breaking at natural boundaries:
    1. Paragraph breaks (\n\n)
    2. Line breaks (\n)
    3. Sentence endings (. )
    4. Words (spaces)
    """
    if not text or len(text) <= chunk_size:
        return [text] if text else []

    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = start + chunk_size

        # If this is the last chunk, take the remainder
        if end >= text_len:
            chunks.append(text[start:].strip())
            break

        # Look for the best natural boundary to split on, searching backwards from `end`
        split_pos = -1
        window = text[start:end]

        for separator in ["\n\n", "\n", ". ", " "]:
            # Search within the last 150 characters of the window to avoid overly short chunks
            search_start = max(0, len(window) - chunk_overlap - 100)
            found = window.rfind(separator, search_start)
            if found != -1:
                # Include the separator in the chunk for punctuation/spaces
                split_pos = start + found + len(separator)
                break

        # If no natural separator is found, hard cut at `end`
        if split_pos == -1 or split_pos <= start:
            split_pos = end

        chunk_text = text[start:split_pos].strip()
        if chunk_text:
            chunks.append(chunk_text)

        # Advance start point, respecting overlap
        start = max(start + 1, split_pos - chunk_overlap)

    return chunks


def detect_section_title(page_text: str, current_section: str = "Preliminary") -> str:
    """
    Identifies the active Constitutional Part, Schedule, or Chapter from the page header.
    Carries over the section title across pages until a new Part/Schedule begins.
    """
    import re
    cleaned_header = page_text[:400].replace("\ufffd", "-")
    # Matches: (Part III.—Fundamental Rights) or PART IV DIRECTIVE PRINCIPLES... or PREAMBLE or SCHEDULE
    m = re.search(
        r'\((Part\s+[IVXLCDM]+[A-Z]?\.?[^\)]*)\)|(PART\s+[IVXLCDM]+[A-Z]?[^\n]*)|(PREAMBLE)|([A-Z\s]+SCHEDULE)',
        cleaned_header
    )
    if m:
        extracted = (m.group(1) or m.group(2) or m.group(3) or m.group(4)).strip()
        extracted = extracted.replace("\ufffd", "-").replace("—", " - ").strip()
        return extracted
    return current_section


def chunk_extracted_pages(
    pages: List[Dict],
    chunk_size: int = 800,
    chunk_overlap: int = 150
) -> List[Dict]:
    """
    Splits each page's text into smaller chunks while attaching rich metadata
    and prepending a contextual 'Sticky Note' header to every chunk.

    Enriched Chunk Format:
    "[Constitution of India | Part III - Fundamental Rights | Page 42]
     21. Protection of life and personal liberty..."
    """
    all_chunks = []
    active_section = "Preamble & Preliminary"

    for page in pages:
        page_num = page["page_number"]
        source = page["source"]
        page_text = page["text"]

        # 1. Update the active constitutional section/part for this page
        active_section = detect_section_title(page_text, current_section=active_section)

        # 2. Split page into overlapping chunks
        raw_chunks = split_text_with_overlap(
            page_text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        for chunk_idx, chunk_content in enumerate(raw_chunks):
            chunk_id = f"p{page_num}_c{chunk_idx}"

            # 3. Prepend contextual sticky-note header to the chunk
            header_tag = f"[Constitution of India | {active_section} | Page {page_num}]"
            enriched_text = f"{header_tag}\n{chunk_content}"

            all_chunks.append({
                "chunk_id": chunk_id,
                "page_number": page_num,
                "section": active_section,
                "source": source,
                "text": enriched_text,
                "raw_text": chunk_content,
                "char_count": len(enriched_text)
            })

    return all_chunks


if __name__ == "__main__":
    import sys
    # Add current script directory to sys.path so 'ingest' can be imported cleanly
    src_dir = os.path.dirname(os.path.abspath(__file__))
    if src_dir not in sys.path:
        sys.path.append(src_dir)

    from ingest import extract_pages_from_pdf

    pdf_path = os.path.join("data", "constitution.pdf")
    print(f"Loading pages from: {pdf_path} ...")
    pages = extract_pages_from_pdf(pdf_path)

    print("Splitting pages into chunks (size=800, overlap=150)...")
    chunks = chunk_extracted_pages(pages, chunk_size=800, chunk_overlap=150)

    print(f"Total pages:  {len(pages)}")
    print(f"Total chunks: {len(chunks)}")
    print(f"Average chunks per page: {len(chunks) / len(pages):.2f}\n")

    # Let's inspect chunks for Article 21 (Page 42)
    p42_chunks = [c for c in chunks if c["page_number"] == 42]
    print("=" * 60)
    print(f"INSPECTING CHUNKS FOR PAGE 42 ({len(p42_chunks)} chunks created)")
    print("=" * 60)
    for c in p42_chunks:
        print(f"\n[{c['chunk_id']}] (Length: {c['char_count']} chars, Page: {c['page_number']})")
        print("-" * 50)
        print(c["text"])

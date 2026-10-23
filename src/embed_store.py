"""
src/embed_store.py
Generates local dense vector embeddings using Sentence Transformers,
constructs a FAISS similarity search index, and persists both index and
metadata to disk for fast reuse.
"""

import os
import sys
import json
import time
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple

# Default local embedding model: 384 dimensions, fast inference, high accuracy
DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
VECTOR_STORE_DIR = "vector_store"
INDEX_FILE = "index.faiss"
METADATA_FILE = "metadata.json"


def get_embedding_model(model_name: str = DEFAULT_MODEL_NAME) -> SentenceTransformer:
    """
    Loads and returns the local SentenceTransformer model.
    Downloads weights on first run, uses local cache thereafter.
    """
    return SentenceTransformer(model_name)


def generate_embeddings(
    texts: List[str],
    model: SentenceTransformer,
    batch_size: int = 64
) -> np.ndarray:
    """
    Encodes a list of text strings into a 2D numpy array of float32 embeddings.
    Normalizes vectors to unit length so that Inner Product (IP) search
    corresponds directly to Cosine Similarity.
    """
    # show_progress_bar gives immediate visual feedback during indexing
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True  # Normalizes each vector to length 1 (L2 norm)
    )

    # FAISS strictly requires float32 data type
    return embeddings.astype(np.float32)


def build_and_save_vector_store(
    chunks: List[Dict],
    store_dir: str = VECTOR_STORE_DIR,
    model_name: str = DEFAULT_MODEL_NAME
) -> None:
    """
    1. Extracts text from all chunks.
    2. Encodes texts into 384-dimensional vectors.
    3. Builds a FAISS IndexFlatIP (Cosine Similarity).
    4. Saves index and metadata to disk.
    """
    os.makedirs(store_dir, exist_ok=True)
    index_path = os.path.join(store_dir, INDEX_FILE)
    metadata_path = os.path.join(store_dir, METADATA_FILE)

    print(f"Loading embedding model: '{model_name}' ...")
    model = get_embedding_model(model_name)

    texts = [c["text"] for c in chunks]
    print(f"Generating embeddings for {len(texts)} chunks...")
    start_time = time.time()
    embeddings = generate_embeddings(texts, model)
    duration = time.time() - start_time
    print(f"Embedding completed in {duration:.2f} seconds ({len(texts)/duration:.1f} chunks/sec).")

    dimension = embeddings.shape[1]  # 384 for all-MiniLM-L6-v2
    print(f"Embedding shape: {embeddings.shape} (Dimensions: {dimension})")

    # IndexFlatIP calculates Inner Product between normalized vectors = Cosine Similarity
    # Scores range from -1.0 to +1.0 (with +1.0 being identical meaning)
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    print(f"Added {index.ntotal} vectors to FAISS index.")

    # Save FAISS index
    faiss.write_index(index, index_path)
    print(f"Saved FAISS index to: {index_path}")

    # Save metadata (texts, page numbers, chunk IDs)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    print(f"Saved metadata to: {metadata_path}")


def load_vector_store(
    store_dir: str = VECTOR_STORE_DIR
) -> Tuple[faiss.Index, List[Dict]]:
    """
    Loads the serialized FAISS index and chunk metadata from disk.
    Fast startup: Avoids re-extracting or re-embedding the PDF!
    """
    index_path = os.path.join(store_dir, INDEX_FILE)
    metadata_path = os.path.join(store_dir, METADATA_FILE)

    if not os.path.exists(index_path) or not os.path.exists(metadata_path):
        raise FileNotFoundError(
            f"Vector store not found in '{store_dir}'. Run indexing first!"
        )

    index = faiss.read_index(index_path)
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return index, metadata


if __name__ == "__main__":
    # Add src/ to path so imports work from anywhere
    src_dir = os.path.dirname(os.path.abspath(__file__))
    if src_dir not in sys.path:
        sys.path.append(src_dir)

    from ingest import extract_pages_from_pdf
    from chunking import chunk_extracted_pages

    pdf_file = os.path.join("data", "constitution.pdf")
    print("=" * 60)
    print("INDEXING PIPELINE DEMONSTRATION")
    print("=" * 60)

    # Step 1: Extract pages
    print("Step 1: Extracting pages from PDF...")
    pages = extract_pages_from_pdf(pdf_file)

    # Step 2: Split into chunks
    print("Step 2: Splitting pages into chunks...")
    chunks = chunk_extracted_pages(pages, chunk_size=800, chunk_overlap=150)

    # Step 3 & 4: Embed & Save into FAISS
    print("Step 3: Building FAISS vector store...")
    build_and_save_vector_store(chunks)

    # Verification: Load back from disk
    print("\nVerifying index persistence...")
    loaded_index, loaded_metadata = load_vector_store()
    print(f" [PASS] Loaded index contains {loaded_index.ntotal} vectors.")
    print(f" [PASS] Loaded metadata contains {len(loaded_metadata)} chunk records.")

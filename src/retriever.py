"""
src/retriever.py
Handles semantic search across the FAISS vector database.
Converts user queries into vectors and retrieves the top-k most relevant chunks.
"""

import os
import sys
import numpy as np
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer

# Add src/ to path so imports work cleanly
src_dir = os.path.dirname(os.path.abspath(__file__))
if src_dir not in sys.path:
    sys.path.append(src_dir)

from embed_store import load_vector_store, get_embedding_model, DEFAULT_MODEL_NAME


class ConstitutionRetriever:
    """
    Manages loading the FAISS index and retrieving top-k relevant constitutional passages.
    """

    def __init__(
        self,
        vector_store_dir: str = "vector_store",
        model_name: str = DEFAULT_MODEL_NAME
    ):
        print(f"Loading FAISS index & metadata from '{vector_store_dir}'...")
        self.index, self.metadata = load_vector_store(vector_store_dir)
        print(f"Loaded index with {self.index.ntotal} vectors.")

        print(f"Loading query embedding model ('{model_name}')...")
        self.model = get_embedding_model(model_name)

    def expand_query(self, query: str) -> str:
        """
        Enriches short or generic queries with domain context.
        For example, 'What is the Preamble?' becomes
        'What is the Preamble? in the Constitution of India'.
        """
        clean = query.strip()
        lower = clean.lower()
        if "constitution" not in lower and "india" not in lower:
            return f"{clean} in the Constitution of India"
        return clean

    def find_article_chunks(self, query: str) -> List[Dict]:
        """
        Detects specific article references in the query (e.g., 'Article 21', 'Article 21A', 'Art 32')
        and finds chunks where that exact constitutional article begins.
        This solves the semantic mismatch where the PDF text writes '21.' instead of 'Article 21.'
        """
        import re
        # Match patterns like 'Article 21', 'article 21A', 'Art. 21', 'Article 51A'
        match = re.search(r'\b(?:article|art\.?)\s*(\d+[A-Za-z]?)', query, re.IGNORECASE)
        if not match:
            return []

        art_num = match.group(1).upper()
        # Look for "21. Protection", "21A. Right", "Article 21." at line starts or word boundaries
        pattern = re.compile(rf'(?:Article\s+|\n|\b){re.escape(art_num)}\.\s+[A-Z]', re.IGNORECASE)

        matched_chunks = []
        for chunk in self.metadata:
            # Exclude schedules or tables where numbers are just list items (look for actual Article clauses)
            text = chunk["text"]
            if pattern.search(text) and "SCHEDULE" not in chunk.get("source", "").upper():
                c = dict(chunk)
                c["similarity_score"] = 0.95  # High priority for exact article match
                matched_chunks.append(c)

        return matched_chunks

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.35
    ) -> List[Dict]:
        """
        Hybrid retrieval:
        1. Exact Article Lookup (if user asked for a specific Article number).
        2. Dense Semantic Vector Search via FAISS.
        3. Merged, deduplicated top_k results.
        """
        if not query.strip():
            return []

        # Step 1: Check for exact article reference
        exact_article_chunks = self.find_article_chunks(query)

        # Step 2: Contextually enrich query for dense semantic recall
        search_query = self.expand_query(query)

        # Step 3: Dense vector search via FAISS
        query_vector = self.model.encode(
            [search_query],
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype(np.float32)

        scores, indices = self.index.search(query_vector, top_k * 2)

        semantic_results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            similarity = float(score)
            if similarity < score_threshold:
                continue
            chunk_data = dict(self.metadata[idx])
            chunk_data["similarity_score"] = round(similarity, 4)
            semantic_results.append(chunk_data)

        # Step 4: Merge results, prioritizing exact article matches first
        seen_ids = set()
        final_results = []

        for c in exact_article_chunks:
            if c["chunk_id"] not in seen_ids:
                seen_ids.add(c["chunk_id"])
                final_results.append(c)

        for c in semantic_results:
            if c["chunk_id"] not in seen_ids:
                seen_ids.add(c["chunk_id"])
                final_results.append(c)

        return final_results[:top_k]


if __name__ == "__main__":
    retriever = ConstitutionRetriever()

    test_questions = [
        "What does Article 21 say about protection of life and personal liberty?",
        "What is the Preamble of the Constitution of India?",
        "What are the Directive Principles of State Policy?",
        "What does the Constitution say about quantum computing and space exploration?"  # Out of syllabus question!
    ]

    print("\n" + "=" * 70)
    print("TESTING RETRIEVAL QUALITY ACROSS QUESTIONS")
    print("=" * 70)

    for q in test_questions:
        print(f"\n Question: '{q}'")
        results = retriever.retrieve(q, top_k=2, score_threshold=0.35)

        if not results:
            print(" [NO RESULTS] No relevant passages met the similarity threshold.")
            continue

        for i, res in enumerate(results, 1):
            print(f"  Result {i}: [Score: {res['similarity_score']:.4f} | Page: {res['page_number']} | ID: {res['chunk_id']}]")
            # Print first 180 characters of snippet
            snippet = res['text'][:180].replace('\n', ' ')
            print(f"   Snippet: {snippet}...\n")

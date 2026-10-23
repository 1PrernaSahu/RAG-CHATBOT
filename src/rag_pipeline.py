"""
src/rag_pipeline.py
The master pipeline that combines:
1. ConstitutionRetriever (FAISS vector search)
2. Groq Generator (Llama/GPT grounded answer generation)
"""

import os
import sys

# Ensure UTF-8 output encoding on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from typing import Dict, List, Optional

# Add src/ to path so imports work cleanly
src_dir = os.path.dirname(os.path.abspath(__file__))
if src_dir not in sys.path:
    sys.path.append(src_dir)

from retriever import ConstitutionRetriever
from generator import generate_answer, stream_answer, contextualize_query, DEFAULT_GROQ_MODEL


class ConstitutionRAG:
    """
    End-to-End Retrieval-Augmented Generation system for the Constitution of India.
    """

    def __init__(
        self,
        vector_store_dir: str = "vector_store",
        groq_model: str = DEFAULT_GROQ_MODEL,
        score_threshold: float = 0.35
    ):
        self.groq_model = groq_model
        self.score_threshold = score_threshold
        print("Initializing Constitution RAG Pipeline...")
        self.retriever = ConstitutionRetriever(vector_store_dir=vector_store_dir)
        print("Pipeline initialized successfully!\n")

    def query(
        self,
        question: str,
        top_k: int = 5,
        chat_history: Optional[List[Dict]] = None
    ) -> Dict:
        """
        Executes complete RAG workflow:
        1. Contextualizes follow-up questions using chat history.
        2. Retrieves relevant passages from FAISS.
        3. Queries Groq LLM for grounded answer.
        """
        clean_q = question.strip()
        if not clean_q:
            return {
                "question": question,
                "answer": "Please ask a question regarding the Constitution of India.",
                "retrieved_chunks": [],
                "model_used": self.groq_model,
                "sufficient_evidence": False
            }

        # Step 0: Conversational Memory (Contextualize follow-up questions)
        search_query = contextualize_query(clean_q, chat_history, model=self.groq_model)

        # Step 1: Retrieval
        retrieved_chunks = self.retriever.retrieve(
            search_query,
            top_k=top_k,
            score_threshold=self.score_threshold
        )

        # Step 2: Quality Check - Guard against zero evidence
        if not retrieved_chunks:
            return {
                "question": question,
                "search_query": search_query,
                "answer": (
                    "Based on the retrieved sections of the Constitution of India, "
                    "I cannot find sufficient information to answer your question. "
                    "Please verify if your question relates to the indexed constitutional text."
                ),
                "retrieved_chunks": [],
                "model_used": self.groq_model,
                "sufficient_evidence": False
            }

        # Step 3 & 4: Generation
        gen_result = generate_answer(
            query=clean_q,
            retrieved_chunks=retrieved_chunks,
            model=self.groq_model
        )

        return {
            "question": clean_q,
            "search_query": search_query,
            "answer": gen_result["answer"],
            "retrieved_chunks": retrieved_chunks,
            "citations": gen_result.get("citations", []),
            "model_used": gen_result.get("model_used", self.groq_model),
            "sufficient_evidence": True,
            "error": gen_result.get("error")
        }

    def stream_query(
        self,
        question: str,
        top_k: int = 5,
        chat_history: Optional[List[Dict]] = None
    ):
        """
        Retrieves context passages using conversational memory,
        then yields text chunks for streaming.
        Returns a tuple: (stream_generator, retrieved_chunks, search_query)
        """
        clean_q = question.strip()
        if not clean_q:
            def empty_stream():
                yield "Please ask a question regarding the Constitution of India."
            return empty_stream(), [], clean_q

        # Step 0: Conversational Memory
        search_query = contextualize_query(clean_q, chat_history, model=self.groq_model)

        # Step 1: Retrieval
        retrieved_chunks = self.retriever.retrieve(
            search_query,
            top_k=top_k,
            score_threshold=self.score_threshold
        )

        if not retrieved_chunks:
            def no_evidence_stream():
                yield (
                    "Based on the retrieved sections of the Constitution of India, "
                    "I cannot find sufficient information to answer your question. "
                    "Please verify if your question relates to the indexed constitutional text."
                )
            return no_evidence_stream(), [], search_query

        # Return generator stream, chunks, and the search query used
        return stream_answer(clean_q, retrieved_chunks, model=self.groq_model), retrieved_chunks, search_query


if __name__ == "__main__":
    rag = ConstitutionRAG()

    test_queries = [
        "What are the Fundamental Rights guaranteed to Indian citizens?",
        "What does Article 21 say about personal liberty?",
        "What is the difference between Fundamental Rights and Directive Principles?",
        "Can a citizen travel faster than the speed of light in space?"  # Hallucination test
    ]

    for q in test_queries:
        print("=" * 70)
        print(f" USER QUESTION: {q}")
        print("=" * 70)

        response = rag.query(q, top_k=3)

        print(f"\n ANSWER ({response['model_used']}):\n")
        print(response["answer"])

        print("\n SUPPORTING CITATIONS:")
        for chunk in response["retrieved_chunks"]:
            print(f"  * Page {chunk['page_number']} | Chunk {chunk['chunk_id']} | Similarity Score: {chunk['similarity_score']:.4f}")
        print("\n")

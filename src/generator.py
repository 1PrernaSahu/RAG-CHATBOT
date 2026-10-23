"""
src/generator.py
Handles connecting to the Groq API and generating grounded, factual answers
using retrieved constitutional passages as context.
"""

import os
import sys

# Ensure UTF-8 output encoding on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from typing import List, Dict, Optional
from dotenv import load_dotenv
from groq import Groq, GroqError, AuthenticationError, RateLimitError, APIConnectionError

# Load environment variables (.env)
load_dotenv()

# Verified active model identifier on Groq
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"
FALLBACK_GROQ_MODEL = "qwen/qwen3.8-27b"

SYSTEM_PROMPT = """You are an authoritative, educational AI assistant specializing in the Constitution of India.

Your primary duty is to answer the user's questions strictly and faithfully based on the retrieved constitutional passages provided below.

Rules to follow:
1. GROUNDING: Base your answer ONLY on the provided context passages. Do NOT extrapolate or assume laws not mentioned in the context.
2. CITATIONS: Whenever you mention a principle, rule, or right, cite the relevant Article number and Page number from the context (e.g., "[Article 21, Page 42]").
3. INSUFFICIENT CONTEXT: If the provided context does not contain sufficient information to answer the question accurately, explicitly state:
   "Based on the retrieved sections of the Constitution of India, I cannot find sufficient information to answer this question."
   Do NOT attempt to invent facts or use ungrounded external assumptions.
4. CLARITY: Present legal concepts in clear, understandable English while preserving the precision of the legal terminology.
5. DISCLAIMER: Remember this is for educational purposes and is not a substitute for professional legal counsel.
"""


def get_groq_client() -> Groq:
    """
    Initializes and returns the Groq client using the GROQ_API_KEY from environment.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is missing! Please ensure it is defined in your .env file."
        )
    return Groq(api_key=api_key)


def contextualize_query(
    query: str,
    chat_history: Optional[List[Dict]] = None,
    model: str = DEFAULT_GROQ_MODEL
) -> str:
    """
    If there is prior conversation history and the latest query is a follow-up
    (e.g. 'Can it be suspended?', 'What about Article 22?'), reformulates it
    into a self-contained standalone search query with full context.
    """
    if not chat_history or len(chat_history) < 2:
        return query

    # Only include user questions and concise assistant summaries
    recent_turns = []
    for msg in chat_history[-4:]:
        role = "User" if msg["role"] == "user" else "Assistant"
        content = msg["content"].strip()
        # Truncate lengthy assistant replies to keep rewrite prompt ultra-fast
        short_content = content[:140].replace("\n", " ") + "..." if len(content) > 140 else content
        recent_turns.append(f"{role}: {short_content}")

    history_str = "\n".join(recent_turns)

    prompt = f"""You are a query reformulation assistant for a Constitution of India chatbot.
Given the recent chat history and a follow-up question, rewrite the question into a clear, standalone search query.
Resolve pronouns (like 'it', 'they', 'this article') into the exact constitutional topic or article being discussed.
Do NOT answer the question. Return ONLY the rewritten question.

Chat History:
{history_str}

Follow-up Question: {query}
Standalone Search Query:"""

    try:
        client = get_groq_client()
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=60,
            temperature=0.0
        )
        rewritten = resp.choices[0].message.content.strip().strip('"')
        return rewritten if rewritten else query
    except Exception:
        # Fallback to original query on any network issue
        return query


def build_rag_prompt(query: str, retrieved_chunks: List[Dict]) -> str:
    """
    Formats retrieved chunks into a clean, structured context block followed by the user question.
    """
    if not retrieved_chunks:
        context_block = "[No relevant passages were found in the document index.]"
    else:
        context_parts = []
        for i, chunk in enumerate(retrieved_chunks, 1):
            page = chunk.get("page_number", "Unknown")
            chunk_id = chunk.get("chunk_id", f"c{i}")
            score = chunk.get("similarity_score", 0.0)
            text = chunk.get("text", "").strip()

            passage_header = f"--- Context Passage {i} (Source Page: {page}, Chunk ID: {chunk_id}, Relevance: {score:.2f}) ---"
            context_parts.append(f"{passage_header}\n{text}")

        context_block = "\n\n".join(context_parts)

    prompt = f"""Context Passages from the Constitution of India:
==================================================
{context_block}
==================================================

User Question: {query}

Please provide a well-structured, grounded answer with source citations based on the context above."""
    return prompt


def generate_answer(
    query: str,
    retrieved_chunks: List[Dict],
    model: str = DEFAULT_GROQ_MODEL,
    temperature: float = 0.2
) -> Dict:
    """
    Sends the grounded prompt to the Groq API and returns the generated answer with metadata.
    
    Returns:
    {
        "answer": str,
        "model_used": str,
        "citations": List[Dict],
        "error": Optional[str]
    }
    """
    try:
        client = get_groq_client()
        user_prompt = build_rag_prompt(query, retrieved_chunks)

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,  # Low temperature (0.2) promotes deterministic, factual answers
            max_tokens=1024
        )

        answer_text = response.choices[0].message.content

        citations = [
            {
                "page_number": c.get("page_number"),
                "chunk_id": c.get("chunk_id"),
                "similarity_score": c.get("similarity_score")
            }
            for c in retrieved_chunks
        ]

        return {
            "answer": answer_text,
            "model_used": model,
            "citations": citations,
            "error": None
        }

    except AuthenticationError:
        return {
            "answer": "Authentication failed: Invalid Groq API Key. Please verify your .env file.",
            "model_used": model,
            "citations": [],
            "error": "AuthenticationError"
        }
    except RateLimitError:
        return {
            "answer": "Rate limit reached on Groq API. Please wait a few moments and try again.",
            "model_used": model,
            "citations": [],
            "error": "RateLimitError"
        }
    except APIConnectionError:
        return {
            "answer": "Network connection error: Unable to reach Groq API servers. Please check your internet connection.",
            "model_used": model,
            "citations": [],
            "error": "APIConnectionError"
        }
    except GroqError as e:
        if model != FALLBACK_GROQ_MODEL:
            print(f"Primary model '{model}' failed: {e}. Retrying with fallback '{FALLBACK_GROQ_MODEL}'...")
            return generate_answer(query, retrieved_chunks, model=FALLBACK_GROQ_MODEL, temperature=temperature)
        return {
            "answer": f"Groq API Error: {str(e)}",
            "model_used": model,
            "citations": [],
            "error": str(e)
        }


def stream_answer(
    query: str,
    retrieved_chunks: List[Dict],
    model: str = DEFAULT_GROQ_MODEL,
    temperature: float = 0.2
):
    """
    Generator function that streams tokens in real-time from the Groq API.
    Yields text chunks as they arrive for a live typewriter effect.
    """
    try:
        client = get_groq_client()
        user_prompt = build_rag_prompt(query, retrieved_chunks)

        response_stream = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
            max_tokens=1024,
            stream=True  # Enables real-time streaming!
        )

        for chunk in response_stream:
            content = chunk.choices[0].delta.content
            if content:
                yield content

    except Exception as e:
        yield f"\n\n[Error while streaming response: {str(e)}]"


if __name__ == "__main__":
    print("Testing Groq Generator with a sample context...")

    sample_chunks = [
        {
            "chunk_id": "p42_c2",
            "page_number": 42,
            "source": "constitution.pdf",
            "similarity_score": 0.88,
            "text": (
                "21. Protection of life and personal liberty.—No person shall be "
                "deprived of his life or personal liberty except according to procedure "
                "established by law.\n"
                "21A. Right to education.—The State shall provide free and compulsory education "
                "to all children of the age of six to fourteen years in such manner as the State may, by law, determine."
            )
        }
    ]

    test_q = "What does Article 21 say, and can personal liberty be restricted?"
    print(f"\nQuestion: {test_q}\n")

    result = generate_answer(test_q, sample_chunks)
    print("=" * 60)
    print(f"ANSWER (Generated by {result['model_used']}):")
    print("=" * 60)
    print(result["answer"])
    print("\nCitations:", result["citations"])

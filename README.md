# 🇮🇳 Constitution of India — Retrieval-Augmented Generation (RAG) Chatbot

An educational, grounded AI chatbot built from scratch in Python to answer user queries about the **Constitution of India** using vector search and Large Language Models.

---

## 🏛️ Project Overview

Large Language Models (LLMs) often hallucinate specific legal clauses or lack up-to-date knowledge of amendments. This project implements a **two-pipeline RAG architecture** (Indexing + Question-Answering) that grounds all answers in the official Constitution PDF published by the **Legislative Department, Ministry of Law and Justice, Government of India**.

Every answer provides:
* Direct constitutional grounding (no hallucinations)
* Specific Article citations and PDF Page references
* Interactive evidence inspection (transparency into retrieved text chunks)

---

## 🛠️ Architecture & Data Flow

```text
[Constitution PDF (402 pages)]
       │
       ▼ (PyMuPDF - fitz)
[Page-by-Page Cleaned Text]
       │
       ▼ (Sliding Window Chunker: 800 chars, 150 overlap)
[1,489 Constitutional Chunks + Metadata]
       │
       ▼ (Local Sentence Transformers: all-MiniLM-L6-v2)
[384-Dimensional Normalized Vectors]
       │
       ▼ (FAISS: IndexFlatIP)
[vector_store/index.faiss + metadata.json]  <-- Persisted to disk for instant startup!

--- USER QUERY LIFECYCLE ---
[User Question]
       │
       ▼ (Same Embedding Model)
[Query Vector] ──► [FAISS Cosine Similarity Search] ──► [Top-K Relevant Chunks]
                                                                │
                                                                ▼
                                      [Grounded Prompt: Context + Instructions + Question]
                                                                │
                                                                ▼
                                              [Groq API: openai/gpt-oss-120b]
                                                                │
                                                                ▼
                                                [Grounded Answer with Citations]
```

---

## 📂 Project Directory Structure

```text
practice2/
├── data/
│   └── constitution.pdf       # Official Constitution of India (May 2024 update, 402 pages)
├── vector_store/
│   ├── index.faiss            # Serialized FAISS Index (1,489 vectors)
│   └── metadata.json          # Chunk texts, page numbers, and chunk IDs
├── src/
│   ├── ingest.py              # PDF extraction & text cleaning
│   ├── chunking.py            # Natural-boundary sliding-window chunker
│   ├── embed_store.py         # Embedding generation & FAISS persistence
│   ├── retriever.py           # Semantic query vector search & relevance thresholding
│   ├── generator.py           # Groq API client with grounded legal system prompt
│   ├── rag_pipeline.py        # Master RAG pipeline connecting retriever + generator
│   └── app.py                 # Streamlit conversational web chat interface
├── tests/
│   ├── verify_env.py          # Environment, dependency & API key verification
│   └── test_evaluation.py     # Benchmark suite (Fact retrieval, negative test, jailbreak defense)
├── .env                       # Local secrets (Groq API Key - git-ignored)
├── .env.example               # Public template for environment variables
├── .gitignore                 # Protects secrets, caches, and large binary artifacts
├── requirements.txt           # Project dependencies
└── README.md                  # Complete technical documentation
```

---

## 🚀 Quickstart Guide

### 1. Prerequisites
* Python 3.10+ (Tested on Python 3.13)
* Free Groq API Key from [console.groq.com/keys](https://console.groq.com/keys)

### 2. Setup Virtual Environment & Install Dependencies
```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Install requirements
pip install -r requirements.txt
```

### 3. Configure Secrets
Create a `.env` file in the root directory:
```env
GROQ_API_KEY=gsk_your_actual_key_here
```

### 4. Run the Streamlit Web Application
```powershell
.\venv\Scripts\streamlit run src\app.py
```
Open your browser at `http://localhost:8501`.

---

## 🧪 Testing & Evaluation Benchmark

Run the automated evaluation suite:
```powershell
.\venv\Scripts\python tests\test_evaluation.py
```
The test suite validates:
1. **Exact Fact Retrieval:** Verifies Article 21 verbatim legal phrasing.
2. **Specific Article Lookup:** Evaluates Article 32 (Supreme Court remedies/writs).
3. **Broad Synthesis:** Verifies Fundamental Duties under Article 51A.
4. **Out-of-Domain Grounding:** Tests rejection of non-constitutional queries (e.g., Bitcoin/Crypto).
5. **Instruction Defense:** Tests resistance against prompt injection / jailbreak attempts.

---

## 💼 Interview Guide: How to Explain this Project

When describing this project to a senior engineer or hiring manager:
1. **The Core Engineering Problem:** Explain that LLMs hallucinate legal details and have context window limitations. RAG solves this by converting retrieval into a high-precision search problem before generation.
2. **Embedding Space Mechanics:** Explain why the query and document chunks must be encoded with the exact same embedding model (`all-MiniLM-L6-v2`) to share a normalized 384-dimensional coordinate space.
3. **Index Optimization:** Highlight that we normalized vectors so that FAISS `IndexFlatIP` computes exact Cosine Similarity with zero trigonometric overhead.
4. **Grounded Prompt Engineering:** Explain how the system prompt strictly instructs the LLM to refuse to answer if retrieved chunks do not provide sufficient evidence, virtually eliminating hallucinations.
5. **Production Next Steps:** Discuss adding hybrid search (BM25 keyword search + FAISS dense retrieval) and cross-encoder re-ranking for enterprise scale.

"""
src/app.py
Interactive Streamlit Web Application for the Constitution of India RAG Chatbot.
Features:
- Conversational chat interface with persistent session history
- Grounded answers with Article and Page number citations
- Expandable evidence drawer showing retrieved passages & similarity scores
- Adjustable retrieval settings (top-k) in sidebar
"""

import os
import sys
import streamlit as st

# Configure page metadata
st.set_page_config(
    page_title="Constitution of India — RAG Chatbot",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Ensure UTF-8 output encoding on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Add src/ to path so imports work cleanly
src_dir = os.path.dirname(os.path.abspath(__file__))
if src_dir not in sys.path:
    sys.path.append(src_dir)

from rag_pipeline import ConstitutionRAG


@st.cache_resource(show_spinner="Loading Constitution FAISS Index & Embedding Models...")
def load_rag_pipeline():
    """
    Caches the RAG pipeline instance so it is loaded once into memory
    and reused across user queries and page re-runs.
    """
    return ConstitutionRAG(vector_store_dir="vector_store")


# Initialize RAG Pipeline
try:
    rag_pipeline = load_rag_pipeline()
    pipeline_loaded = True
except Exception as e:
    pipeline_loaded = False
    pipeline_error = str(e)


# --- SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.title("⚖️ Constitution RAG")
    st.markdown(
        """
        An **educational AI assistant** grounded strictly in the **Constitution of India** (as updated May 2024).
        """
    )
    st.divider()

    st.subheader("⚙️ Retrieval Parameters")
    top_k = st.slider(
        "Context Passages (Top-K):",
        min_value=2,
        max_value=8,
        value=5,
        help="Number of most relevant constitutional chunks to supply to the LLM."
    )

    st.divider()
    st.subheader("📊 System Specs")
    st.markdown(
        """
        - **Source:** *Legislative Dept., Govt. of India*
        - **Document Pages:** 402 pages
        - **Indexed Chunks:** 1,489 chunks
        - **Embedding Model:** `all-MiniLM-L6-v2` (Local)
        - **Vector Store:** FAISS (`IndexFlatIP`)
        - **LLM Engine:** Groq (`openai/gpt-oss-120b`)
        """
    )

    st.divider()
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.caption("⚠️ For educational and informational purposes only. Not legal advice.")


# --- MAIN CHAT INTERFACE ---
st.title("🇮🇳 Constitution of India — AI Assistant")
st.markdown(
    "Ask any question about Fundamental Rights, Articles, Duties, or Constitutional Principles. "
    "Every answer is **grounded in retrieved text** with verified citations."
)

# Sample question pills
st.markdown("**Try asking:**")
sample_cols = st.columns(4)
sample_questions = [
    "What does Article 21 say?",
    "What are Fundamental Rights?",
    "What is the Preamble?",
    "Directive Principles vs Rights?"
]

clicked_sample = None
for i, col in enumerate(sample_cols):
    if col.button(sample_questions[i], key=f"sample_{i}", use_container_width=True):
        clicked_sample = sample_questions[i]

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Namaste! I am your **Constitution of India AI Guide**.\n\n"
                "I search through all 402 pages and 1,489 indexed sections of the official Constitution "
                "to provide accurate, grounded answers with citations. How can I assist you today?"
            ),
            "retrieved_chunks": []
        }
    ]

# Display existing conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # If assistant response has retrieved chunks, show them in an expander
        if msg.get("retrieved_chunks"):
            with st.expander("📚 View Supporting Constitutional Passages (Evidence)"):
                for idx, chunk in enumerate(msg["retrieved_chunks"], 1):
                    score = chunk.get("similarity_score", 0.0)
                    page = chunk.get("page_number", "?")
                    chunk_id = chunk.get("chunk_id", "?")
                    st.markdown(
                        f"**Passage {idx}** — *Page {page}* | *Chunk: `{chunk_id}`* | *Similarity Score: `{score:.4f}`*"
                    )
                    st.code(chunk.get("text", ""), language="text")
                    st.divider()

# Handle new user input (either from chat input or clicked sample pill)
user_prompt = st.chat_input("Ask a question about the Constitution of India...")
if clicked_sample:
    user_prompt = clicked_sample

if user_prompt:
    if not pipeline_loaded:
        st.error(f"Cannot process question: RAG pipeline failed to load: {pipeline_error}")
    else:
        # 1. Display user query
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        # 2. Generate grounded response with real-time typewriter streaming
        with st.chat_message("assistant"):
            stream_gen, retrieved_chunks, search_query = rag_pipeline.stream_query(
                user_prompt,
                top_k=top_k,
                chat_history=st.session_state.messages
            )
            # st.write_stream renders tokens live word-by-word like ChatGPT!
            answer = st.write_stream(stream_gen)

            # Display supporting evidence in an expander
            if retrieved_chunks:
                with st.expander("📚 View Supporting Constitutional Passages (Evidence)", expanded=False):
                    if search_query != user_prompt:
                        st.info(f"🧠 **Conversational Memory:** Rewrote query as: *\"{search_query}\"*")
                    for idx, chunk in enumerate(retrieved_chunks, 1):
                        score = chunk.get("similarity_score", 0.0)
                        page = chunk.get("page_number", "?")
                        chunk_id = chunk.get("chunk_id", "?")
                        section = chunk.get("section", "")
                        sec_str = f" | *{section}*" if section else ""
                        st.markdown(
                            f"**Passage {idx}** — *Page {page}*{sec_str} | *Chunk: `{chunk_id}`* | *Similarity: `{score:.4f}`*"
                        )
                        st.code(chunk.get("text", ""), language="text")
                        st.divider()

            # Save assistant response to session state
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "retrieved_chunks": retrieved_chunks,
                "search_query": search_query
            })

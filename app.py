import os
import streamlit as st
import time
from dotenv import load_dotenv

# Import our custom modules
from ingest import load_and_chunk_corpus
from vector_store import upsert_chunks
from rag import execute_rag_pipeline

# Page Configuration
st.set_page_config(
    page_title="ASUS TUF Gaming F16 (2025) RAG Chatbot",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling using CSS injections
st.markdown("""
<style>
    /* Import modern Google font */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=JetBrains+Mono:wght@400;600&display=swap');
    
    /* Apply styles globally */
    * {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Title customization */
    .title-container {
        background: linear-gradient(135deg, #FF6F00 0%, #FF9F00 50%, #FFD000 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 3rem;
        margin-bottom: 0.5rem;
    }
    
    .subtitle-container {
        font-size: 1.1rem;
        color: #888888;
        margin-bottom: 2rem;
    }
    
    /* Answer Card (Glassmorphic) */
    .answer-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 24px;
        margin-top: 20px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
    }
    
    /* Prompt label */
    .prompt-header {
        font-weight: 600;
        font-size: 1.4rem;
        color: #FFA000;
        margin-bottom: 8px;
    }
    
    /* Metadata Badge styling */
    .badge {
        display: inline-block;
        padding: 4px 10px;
        font-size: 0.8rem;
        font-weight: 600;
        border-radius: 20px;
        margin-right: 8px;
        margin-bottom: 8px;
        text-transform: uppercase;
        font-family: 'JetBrains Mono', monospace;
    }
    
    .badge-intent {
        background-color: #3f51b5;
        color: white;
    }
    
    .badge-file {
        background-color: #009688;
        color: white;
    }
    
    .badge-score {
        background-color: #ff5722;
        color: white;
    }
    
    /* Custom button styling */
    div.stButton > button {
        background: linear-gradient(135deg, #FF6F00 0%, #FF9F00 100%);
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 10px 24px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(255, 111, 0, 0.4) !important;
    }
    
    div.stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(255, 111, 0, 0.6) !important;
    }
</style>
""", unsafe_allow_html=True)

# Load environment variables from absolute path .env
current_dir = os.path.dirname(os.path.abspath(__file__))
dotenv_path = os.path.join(current_dir, ".env")
load_dotenv(dotenv_path=dotenv_path)

# Setup default keys in state if they exist in environment
if "gemini_key" not in st.session_state:
    st.session_state.gemini_key = os.getenv("GEMINI_API_KEY", "")
if "pinecone_key" not in st.session_state:
    st.session_state.pinecone_key = os.getenv("PINECONE_API_KEY", "")

# --- SIDEBAR: Credentials and Control ---
with st.sidebar:
    st.image("https://img.icons8.com/nolan/128/laptop.png", width=100)
    st.markdown("### Configuration Panel")
    
    # Text inputs for keys
    st.session_state.gemini_key = st.text_input(
        "OpenRouter API Key",
        value=st.session_state.gemini_key,
        type="password",
        help="Paste your OpenRouter API Key starting with sk-or-v1-"
    )
    
    st.session_state.pinecone_key = st.text_input(
        "Pinecone API Key",
        value=st.session_state.pinecone_key,
        type="password",
        help="Paste your Pinecone API Key"
    )
    
    # Save the keys to current environment so subprocesses/modules can access them
    os.environ["GEMINI_API_KEY"] = st.session_state.gemini_key
    os.environ["PINECONE_API_KEY"] = st.session_state.pinecone_key
    
    st.markdown("---")
    
    # Ingestion Pipeline Trigger
    st.markdown("### Document Ingestion")
    st.markdown("Click below to read, chunk, embed, and index files from the `corpus/` folder into your Pinecone namespace.")
    
    if st.button("🚀 Run Ingestion & Indexing"):
        # Validate keys are provided
        if not st.session_state.gemini_key or not st.session_state.pinecone_key:
            st.error("Please provide both OpenRouter and Pinecone API keys first!")
        else:
            with st.spinner("Ingesting and indexing corpus documents..."):
                try:
                    start_time = time.time()
                    # 1. Load and chunk
                    st.write("📖 Extracting text and creating chunks...")
                    chunks = load_and_chunk_corpus()
                    st.info(f"Loaded {len(chunks)} chunks from corpus files.")
                    
                    # 2. Embed and upload to Pinecone
                    st.write("🧠 Generating embeddings and uploading to Pinecone...")
                    upsert_chunks(chunks)
                    
                    elapsed = time.time() - start_time
                    st.success(f"Success! Indexed {len(chunks)} chunks in {elapsed:.2f} seconds.")
                except Exception as e:
                    st.error(f"Error during ingestion pipeline: {e}")
                    
    st.markdown("---")
    
    # Query configuration
    st.markdown("### Retrieval Settings")
    top_k = st.slider("Number of retrieved chunks (k)", min_value=1, max_value=8, value=4)
    st.caption("Lower values retrieve fewer source chunks; higher values provide more context to the LLM.")
    
    st.markdown("### LLM Model Settings")
    selected_model = st.selectbox(
        "Select OpenRouter Model",
        options=[
            "openrouter/free",
            "meta-llama/llama-3.3-70b-instruct:free",
            "meta-llama/llama-3.2-3b-instruct:free",
            "google/gemini-2.5-flash"
        ],
        index=0,
        help="Select 'openrouter/free' or another free model if your API key has a $0 balance to avoid 402/429 rate limit errors!"
    )

# --- MAIN APP LAYOUT ---

# Header Section
st.markdown('<div class="title-container">ASUS TUF Gaming F16 (2025) RAG Chatbot</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle-container">Ask questions, retrieve matches, and get grounded generation responses</div>', unsafe_allow_html=True)

# Connection Status Check
if not st.session_state.gemini_key or st.session_state.gemini_key == "your_gemini_api_key_here":
    st.warning("⚠️ OpenRouter API Key is missing. Please add it in the sidebar to use the chatbot.")
if not st.session_state.pinecone_key or st.session_state.pinecone_key == "your_pinecone_api_key_here":
    st.warning("⚠️ Pinecone API Key is missing. Please add it in the sidebar to enable search features.")

# Quick Suggestions Section
st.markdown("### 💡 Quick Suggestion Questions:")
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("What processor configurations are available?"):
        st.session_state.query_input = "What processor configurations are available?"
    if st.button("Is the fan noise loud during gaming?"):
        st.session_state.query_input = "Is the fan noise loud during gaming?"
with col2:
    if st.button("How do I upgrade the RAM on this laptop?"):
        st.session_state.query_input = "How do I upgrade the RAM on this laptop?"
    if st.button("Can I expand the storage of the laptop?"):
        st.session_state.query_input = "Can I expand the storage of the laptop?"
with col3:
    if st.button("Does it meet military durability standards?"):
        st.session_state.query_input = "Does it meet military durability standards?"
    if st.button("Who won the IPL 2025 tournament?"):
        st.session_state.query_input = "Who won the IPL 2025 tournament?"

# Main text input box for user question
query = st.text_input(
    "Enter your question about ASUS TUF Gaming F16 (2025):",
    value=st.session_state.get("query_input", ""),
    placeholder="e.g. Can I upgrade the RAM? / What GPU does this laptop have?"
)

# Ask Button
if st.button("🔍 Ask Chatbot"):
    if not query.strip():
        st.warning("Please enter a question first.")
    elif not st.session_state.gemini_key or not st.session_state.pinecone_key:
        st.error("Please configure your API keys in the sidebar before asking!")
    else:
        with st.spinner("Classifying intent and retrieving context..."):
            try:
                # Execute pipeline
                answer, retrieved_chunks, intent, meta = execute_rag_pipeline(query, top_k=top_k, model_name=selected_model)
                
                # Display Answer in a premium card
                st.markdown('<div class="prompt-header">Generated Response:</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="answer-card">{answer}</div>', unsafe_allow_html=True)
                
                # Expandable pipeline diagnostic details
                with st.expander("🛠️ System Diagnostic & RAG Execution Trace"):
                    st.markdown("### Pipeline Execution Information")
                    st.write(f"**Target Model:** `{selected_model}`")
                    st.write(f"**Detected Intent:** `{intent}`")
                    st.write(f"**Highest Similarity Score:** `{meta.get('highest_similarity_score', 0.0):.4f}`")
                    st.write(f"**Metadata Filter Applied:** `{meta.get('retrieval_filtered', False)}`")
                    st.write(f"**Fallback Retrieval Triggered:** `{meta.get('fallback_retrieval_triggered', False)}`")
                    
                    st.markdown("### Retrieved Context Chunks (Top-K)")
                    if not retrieved_chunks:
                        st.info("No chunks were retrieved (either out of scope, or zero matching results).")
                    else:
                        for idx, chunk in enumerate(retrieved_chunks):
                            score = chunk.get("score", 0.0)
                            source = chunk.get("source_file", "unknown")
                            chunk_text = chunk.get("text", "")
                            doc_type = chunk.get("metadata", {}).get("document_type", "unknown")
                            
                            st.markdown(f"#### Chunk {idx + 1}")
                            # Badges
                            st.markdown(f"""
                            <span class="badge badge-file">📄 File: {source}</span>
                            <span class="badge badge-intent">📂 Doc Type: {doc_type}</span>
                            <span class="badge badge-score">🎯 Similarity Score: {score:.4f}</span>
                            """, unsafe_allow_html=True)
                            
                            # Content box
                            st.code(chunk_text, language="text")
                            st.markdown("---")
            except Exception as e:
                st.error(f"Failed to execute pipeline: {e}")

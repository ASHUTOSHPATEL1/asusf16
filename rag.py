import os
import requests
import json
from dotenv import load_dotenv

# Import components from other modules
from intent_classifier import classify_query_intent
from vector_store import retrieve_similar_chunks

# Load environment variables from absolute path .env
current_dir = os.path.dirname(os.path.abspath(__file__))
dotenv_path = os.path.join(current_dir, ".env")
load_dotenv(dotenv_path=dotenv_path)

# Map detected intents to document types
INTENT_FILTER_MAP = {
    "specifications": ["spec", "faq"],
    "setup/how_to": ["manual", "faq"],
    "troubleshooting": ["manual", "faq"],
    "reviews_opinions": ["review"],
    "general_product_question": None, # Search all
    "out_of_scope": None
}

SIM_THRESHOLD = 0.20  # Minimum similarity score required to trust documents

def execute_rag_pipeline(query, top_k=4, model_name=None):
    """
    Coordinates the complete RAG pipeline:
    1. Detects query intent.
    2. Applies guardrails (out of scope, low similarity scores).
    3. Retrieves filtered or fallback chunks from Pinecone.
    4. Call LLM for grounded generation.
    Returns: (answer_string, retrieved_chunks_list, detected_intent, pipeline_metadata_dict)
    """
    # 1. Intent Detection
    print(f"\n--- Starting RAG Pipeline for query: '{query}' ---")
    intent = classify_query_intent(query, model_name=model_name)
    print(f"Detected Intent: {intent}")
    
    pipeline_metadata = {
        "detected_intent": intent,
        "fallback_retrieval_triggered": False,
        "highest_similarity_score": 0.0,
        "retrieval_filtered": False
    }

    # 2. Guardrail: Out of Scope
    if intent == "out_of_scope":
        print("Guardrail: Query classified as 'out_of_scope'. Bypassing retrieval and generation.")
        return "I don't know from the available info.", [], intent, pipeline_metadata

    # 3. Document Retrieval
    doc_types = INTENT_FILTER_MAP.get(intent)
    pipeline_metadata["retrieval_filtered"] = doc_types is not None
    
    # Try retrieval with intent filter
    retrieved_chunks = retrieve_similar_chunks(query, top_k=top_k, doc_type_filter=doc_types)
    
    # Check if we got anything. If not, or if similarity is extremely low, fall back to searching all files.
    max_score = max([c["score"] for c in retrieved_chunks]) if retrieved_chunks else 0.0
    pipeline_metadata["highest_similarity_score"] = max_score
    
    if (not retrieved_chunks or max_score < 0.15) and doc_types is not None:
        print("Fallback: Filtered retrieval returned low relevance. Retrying without filters...")
        retrieved_chunks = retrieve_similar_chunks(query, top_k=top_k, doc_type_filter=None)
        max_score = max([c["score"] for c in retrieved_chunks]) if retrieved_chunks else 0.0
        pipeline_metadata["highest_similarity_score"] = max_score
        pipeline_metadata["fallback_retrieval_triggered"] = True

    # 4. Guardrail: Relevance/Similarity Threshold
    if not retrieved_chunks or max_score < SIM_THRESHOLD:
        print(f"Guardrail: Highest similarity score ({max_score:.4f}) is below threshold ({SIM_THRESHOLD}). Rejecting query.")
        return "I don't know from the available info.", retrieved_chunks, intent, pipeline_metadata

    # 5. Grounded Generation
    try:
        answer = generate_grounded_answer(query, retrieved_chunks, model_name=model_name)
        return answer, retrieved_chunks, intent, pipeline_metadata
    except Exception as e:
        print(f"Error during grounded generation: {e}")
        raise e

def generate_grounded_answer(query, retrieved_chunks, model_name=None):
    """
    Builds the system prompt and context, then calls OpenRouter model.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not set. Please update the .env file.")

    if not model_name:
        model_name = os.getenv("LLM_MODEL", "openrouter/free")

    # Format context block
    context_str = ""
    for i, chunk in enumerate(retrieved_chunks):
        context_str += f"--- Document Chunk {i+1} (Source: {chunk['source_file']}, Similarity Score: {chunk['score']:.4f}) ---\n"
        context_str += f"{chunk['text']}\n\n"

    # Enforce strict system prompt
    system_prompt = """You are an assistant for ASUS TUF Gaming F16 (2025). Answer only from the provided context. Use only the retrieved documents to answer questions. Do not invent specifications, features, instructions, or opinions that are not explicitly present in the context. If the answer is not present in the retrieved context or the question is unrelated to ASUS TUF Gaming F16 (2025), reply exactly:

"I don't know from the available info." """

    user_prompt = f"""Retrieved Context:
{context_str}

User Question:
{query}

Answer:"""

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8501",
        "X-Title": "ASUS TUF RAG Chatbot"
    }

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.0  # Keep response grounded and factual
    }

    import time
    max_retries = 5
    delay = 3
    for attempt in range(max_retries):
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=15
            )
            
            # Handle rate limiting (429)
            if response.status_code == 429:
                if attempt == max_retries - 1:
                    print("Error: Max retries reached with 429 rate limit in grounded generation.")
                    raise RuntimeError("OpenRouter free-tier rate limit exceeded. Please wait a few seconds and try again.")
                print(f"Rate limited (429). Retrying in {delay} seconds (Attempt {attempt+1}/{max_retries})...")
                time.sleep(delay)
                delay *= 2
                continue
                
            response.raise_for_status()
            result = response.json()
            answer = result["choices"][0]["message"]["content"].strip()
            return answer
            
        except Exception as e:
            if attempt == max_retries - 1:
                # If it's already a rate limit error, propagate our custom message
                if "rate limit exceeded" in str(e):
                    raise e
                print(f"Error in grounded generation request: {e}")
                raise RuntimeError(f"Grounded generation failed: {e}")
            time.sleep(delay)
            delay *= 2

if __name__ == "__main__":
    # Test RAG pipeline locally (make sure PINECONE_API_KEY and GEMINI_API_KEY are configured in .env)
    print("Testing RAG Pipeline...")
    query = "What is the GPU model in ASUS TUF Gaming F16 (2025)?"
    try:
        ans, chunks, intent, meta = execute_rag_pipeline(query)
        print(f"\nAnswer: {ans}")
        print(f"Retrieved {len(chunks)} chunks.")
    except Exception as e:
        print(f"RAG execution failed: {e}")

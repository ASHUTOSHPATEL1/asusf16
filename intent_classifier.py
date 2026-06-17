import os
import json
import requests
from dotenv import load_dotenv

# Load environment variables from absolute path .env
current_dir = os.path.dirname(os.path.abspath(__file__))
dotenv_path = os.path.join(current_dir, ".env")
load_dotenv(dotenv_path=dotenv_path)

# Valid intent classifications
VALID_INTENTS = [
    "specifications",
    "setup/how_to",
    "troubleshooting",
    "reviews_opinions",
    "general_product_question",
    "out_of_scope"
]

def classify_query_intent(query, model_name=None):
    """
    Classifies a user query using OpenRouter models.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not set. Please update the .env file.")

    if not model_name:
        model_name = os.getenv("LLM_MODEL", "openrouter/free")

    # Detailed system prompt for intent classification
    prompt = f"""You are a query classifier for an ASUS TUF Gaming F16 (2025) laptop assistant.
Your sole job is to classify the user's query into EXACTLY one of these 6 intents:
- specifications
- setup/how_to
- troubleshooting
- reviews_opinions
- general_product_question
- out_of_scope

Guidelines:
1. 'specifications': Asking for technical specs, models, dimensions, weights, ports, parts, hardware info, capabilities.
2. 'setup/how_to': Asking how to install, set up, upgrade, boot, enter BIOS, update, change modes, or configure components.
3. 'troubleshooting': Asking about an issue, error, sound, heat problem, screen flicker, freeze, or fix for abnormal behavior.
4. 'reviews_opinions': Asking about user sentiment, reviews, pros/cons, gaming experiences, if it's "good" or "bad", or rating.
5. 'general_product_question': General, vague, or high-level questions about the laptop that don't fit the above but are related.
6. 'out_of_scope': The question is completely unrelated to laptops or the ASUS TUF Gaming F16 (2025) (e.g. asking about sports, weather, history, programming, other unrelated brands, etc.).

Examples:
- "What processor does this laptop have?" -> specifications
- "How do I enter BIOS?" -> setup/how_to
- "My screen is flickering, how to fix it?" -> troubleshooting
- "Is this laptop good for gaming?" -> reviews_opinions
- "Can I upgrade RAM?" -> general_product_question
- "Who won IPL 2025?" -> out_of_scope
- "What is the weather in Delhi?" -> out_of_scope
- "Tell me about ASUS TUF Gaming F16 (2025) battery life according to reviews." -> reviews_opinions

Query to classify: "{query}"

Respond with ONLY the exact intent name from the list above. Do not include formatting, markdown, punctuation, or any explanation. Output must be exactly one of the 6 words/phrases.
"""

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8501", # Optional, but good practice for OpenRouter identification
        "X-Title": "ASUS TUF RAG Chatbot"
    }

    payload = {
        "model": model_name,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0  # Deterministic response
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
                timeout=10
            )
            
            # Handle rate limiting (429)
            if response.status_code == 429:
                if attempt == max_retries - 1:
                    print("Error: Max retries reached with 429 rate limit in intent classifier. Falling back to 'general_product_question'.")
                    return "general_product_question"
                print(f"Rate limited (429). Retrying in {delay} seconds (Attempt {attempt+1}/{max_retries})...")
                time.sleep(delay)
                delay *= 2
                continue
                
            response.raise_for_status()
            
            result = response.json()
            raw_intent = result["choices"][0]["message"]["content"].strip().lower()
            
            # Clean up any potential markdown formatting (e.g. `specifications`)
            clean_intent = raw_intent.replace("`", "").replace("'", "").replace('"', "").strip()
            
            if clean_intent in VALID_INTENTS:
                return clean_intent
            
            # Fuzzy fallback mapping if the model added extra characters
            for valid in VALID_INTENTS:
                if valid in clean_intent:
                    return valid
                    
            print(f"Warning: Classifier returned non-standard intent: '{clean_intent}'. Defaulting to 'general_product_question'.")
            return "general_product_question"
            
        except Exception as e:
            if attempt == max_retries - 1:
                # Non-silent error handling: Print the traceback/info for interview transparency
                print(f"Error in intent classifier API call: {e}. Falling back to 'general_product_question'.")
                return "general_product_question"
            time.sleep(delay)
            delay *= 2
            
    # Return fallback intent so that execution continues
    return "general_product_question"

if __name__ == "__main__":
    # Test classifier locally
    print("Testing Intent Classifier...")
    test_queries = [
        "What processor does this laptop have?",
        "How do I enter BIOS?",
        "Is this laptop good for gaming?",
        "Can I upgrade RAM?",
        "Who won IPL 2025?"
    ]
    for q in test_queries:
        print(f"Query: '{q}' -> Intent: '{classify_query_intent(q)}'")

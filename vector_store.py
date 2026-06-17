import os
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

# Load environment variables from absolute path .env
current_dir = os.path.dirname(os.path.abspath(__file__))
dotenv_path = os.path.join(current_dir, ".env")
load_dotenv(dotenv_path=dotenv_path)

INDEX_NAME = "asus-tuf-gaming-f16-2025"
NAMESPACE = "asus_tuf_gaming_f16_2025"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Global placeholder to cache the sentence transformer model
_model = None

def get_embedding_model():
    """
    Lazy loads and caches the SentenceTransformer model to save memory/startup time.
    """
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        print(f"Loading embedding model '{EMBEDDING_MODEL_NAME}'...")
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model

def get_pinecone_client():
    """
    Initializes and returns a Pinecone client using the environment variable.
    """
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key or api_key == "your_pinecone_api_key_here":
        raise ValueError("PINECONE_API_KEY environment variable is not set. Please update the .env file.")
    return Pinecone(api_key=api_key)

def initialize_index():
    """
    Connects to the Pinecone index. Creates it if it doesn't already exist.
    all-MiniLM-L6-v2 produces vectors of 384 dimensions.
    """
    pc = get_pinecone_client()
    
    # Pinecone indices must consist of lower case alphanumeric characters or '-'
    if INDEX_NAME not in [idx.name for idx in pc.list_indexes()]:
        print(f"Creating new Pinecone index: {INDEX_NAME}...")
        try:
            pc.create_index(
                name=INDEX_NAME,
                dimension=384,  # Model outputs 384 dimensions
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"  # Standard AWS starter region
                )
            )
            print("Index created successfully.")
        except Exception as e:
            print(f"Error creating Pinecone index: {e}")
            raise e
    else:
        print(f"Pinecone index '{INDEX_NAME}' already exists.")
        
    return pc.Index(INDEX_NAME)

def upsert_chunks(chunks_with_metadata):
    """
    Generates embeddings for a list of document chunks and uploads them to Pinecone.
    """
    model = get_embedding_model()
    index = initialize_index()
    
    print(f"Generating embeddings for {len(chunks_with_metadata)} chunks...")
    vectors = []
    
    for idx, item in enumerate(chunks_with_metadata):
        text = item["text"]
        metadata = item["metadata"]
        
        # We store the raw text directly in the metadata so we can retrieve it
        # without needing a secondary database lookup.
        metadata["text"] = text
        
        # Generate the embedding vector
        embedding = model.encode(text).tolist()
        
        vectors.append({
            "id": metadata["chunk_id"],
            "values": embedding,
            "metadata": metadata
        })
        
    # Upsert vectors in batches to stay within Pinecone payload limits
    batch_size = 100
    print(f"Upserting vectors in batches of {batch_size} to Pinecone namespace '{NAMESPACE}'...")
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i + batch_size]
        try:
            index.upsert(vectors=batch, namespace=NAMESPACE)
            print(f"Uploaded batch {i // batch_size + 1}/{((len(vectors) - 1) // batch_size) + 1}")
        except Exception as e:
            print(f"Error during upsert batch starting at {i}: {e}")
            raise e
            
    print("Ingestion and indexing complete.")

def retrieve_similar_chunks(query, top_k=4, doc_type_filter=None):
    """
    Queries Pinecone for the top_k most similar chunks matching the query.
    Optionally applies a metadata filter on document_type.
    """
    index = initialize_index()
    model = get_embedding_model()
    
    # Generate query embedding vector
    query_vector = model.encode(query).tolist()
    
    # Construct filter dictionary for Pinecone metadata
    filter_dict = {}
    if doc_type_filter:
        if isinstance(doc_type_filter, list):
            # If a list of document types is provided, use Pinecone's $in operator
            filter_dict = {"document_type": {"$in": doc_type_filter}}
        else:
            # If a single string is provided, filter for exact match
            filter_dict = {"document_type": doc_type_filter}
            
    try:
        print(f"Querying Pinecone for: '{query}' (Filter: {doc_type_filter})...")
        response = index.query(
            namespace=NAMESPACE,
            vector=query_vector,
            top_k=top_k,
            include_metadata=True,
            filter=filter_dict if filter_dict else None
        )
        
        results = []
        for match in response.get("matches", []):
            meta = match.get("metadata", {})
            results.append({
                "text": meta.get("text", ""),
                "source_file": meta.get("source_file", "unknown"),
                "score": match.get("score", 0.0),
                "metadata": meta
            })
        return results
    except Exception as e:
        print(f"Pinecone query failed: {e}")
        raise e

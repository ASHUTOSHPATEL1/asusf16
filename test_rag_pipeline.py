import os
import sys

def test_pipeline():
    print("=== Verification Script for RAG Chatbot ===")
    
    # 1. Test Imports
    print("\n1. Testing module imports...")
    try:
        import ingest
        import vector_store
        import intent_classifier
        import rag
        print("[OK] All local modules (ingest, vector_store, intent_classifier, rag) imported successfully!")
    except Exception as e:
        print(f"[ERROR] Import failed: {e}")
        sys.exit(1)
        
    # 2. Test Ingestion Local Parsing
    print("\n2. Testing document parsing and chunking...")
    corpus_dir = "corpus"
    if not os.path.exists(corpus_dir):
        print(f"[ERROR] Corpus directory '{corpus_dir}' does not exist.")
        sys.exit(1)
        
    try:
        # Change current working directory to run as if we're in the project folder
        orig_cwd = os.getcwd()
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        
        chunks = ingest.load_and_chunk_corpus(corpus_dir=corpus_dir)
        print(f"[OK] Document parsing successful! Generated {len(chunks)} total chunks.")
        
        # Verify metadata keys
        required_keys = {"source_file", "document_type", "chunk_id"}
        all_keys_valid = True
        for chunk in chunks:
            if not required_keys.issubset(chunk["metadata"].keys()):
                all_keys_valid = False
                break
        
        if all_keys_valid:
            print("[OK] Chunk metadata structures are valid.")
        else:
            print("[ERROR] Some chunk metadata keys are missing.")
            sys.exit(1)
            
        # Revert CWD
        os.chdir(orig_cwd)
            
    except Exception as e:
        print(f"[ERROR] Ingestion test failed: {e}")
        sys.exit(1)

    print("\n=== Validation Completed: Pipeline Modules & local parser are OK! ===")

if __name__ == "__main__":
    test_pipeline()

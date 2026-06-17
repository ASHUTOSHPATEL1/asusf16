import os
from pypdf import PdfReader

# Map file names to their corresponding document types
DOC_TYPE_MAP = {
    "official_specs.txt": "spec",
    "user_manual.pdf": "manual",
    "faq.txt": "faq",
    "reviews.txt": "review"
}

def extract_text_from_txt(file_path):
    """
    Reads a text file and returns its content.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"Error reading text file {file_path}: {e}")
        raise e

def extract_text_from_pdf(file_path):
    """
    Reads a PDF file using pypdf and returns the concatenated text from all pages.
    """
    try:
        reader = PdfReader(file_path)
        text_content = []
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                text_content.append(page_text)
        return "\n".join(text_content)
    except Exception as e:
        print(f"Error reading PDF file {file_path}: {e}")
        raise e

def chunk_text(text, chunk_size=500, overlap=50):
    """
    Splits text into chunks of approximately chunk_size words with a specified overlap.
    This is simple, beginner-friendly, and avoids complex tokenizer dependencies.
    """
    words = text.split()
    chunks = []
    
    # Calculate step size
    step = chunk_size - overlap
    if step <= 0:
        raise ValueError("Chunk size must be greater than overlap.")
        
    for i in range(0, len(words), step):
        # Slice words
        chunk_words = words[i:i + chunk_size]
        chunk_str = " ".join(chunk_words)
        chunks.append(chunk_str)
        
        # Stop if we've reached the end of the text
        if i + chunk_size >= len(words):
            break
            
    return chunks

def load_and_chunk_corpus(corpus_dir=None):
    """
    Loads all files from the corpus directory, extracts text, chunks it,
    and returns a list of dictionaries with text and metadata.
    """
    all_chunks = []
    
    if corpus_dir is None:
        # Automatically resolve to the 'corpus' folder next to this script file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        corpus_dir = os.path.join(current_dir, "corpus")
        
    if not os.path.exists(corpus_dir):
        raise FileNotFoundError(f"Corpus directory '{corpus_dir}' not found.")
        
    for filename in os.listdir(corpus_dir):
        file_path = os.path.join(corpus_dir, filename)
        
        # Determine document type based on mapping
        doc_type = DOC_TYPE_MAP.get(filename)
        if not doc_type:
            print(f"Skipping unknown file in corpus: {filename}")
            continue
            
        print(f"Processing file: {filename} (Type: {doc_type})...")
        
        # Extract text based on file extension
        if filename.endswith(".pdf"):
            text = extract_text_from_pdf(file_path)
        elif filename.endswith(".txt"):
            text = extract_text_from_txt(file_path)
        else:
            print(f"Unsupported file format: {filename}")
            continue
            
        # Segment into chunks
        chunks = chunk_text(text, chunk_size=500, overlap=50)
        print(f"Split {filename} into {len(chunks)} chunks.")
        
        # Construct chunk metadata list
        for index, chunk_text_content in enumerate(chunks):
            chunk_id = f"{filename.replace('.', '_')}_chunk_{index}"
            all_chunks.append({
                "text": chunk_text_content,
                "metadata": {
                    "source_file": filename,
                    "document_type": doc_type,
                    "chunk_id": chunk_id
                }
            })
            
    return all_chunks

if __name__ == "__main__":
    # Test execution to verify ingestion
    print("Testing Ingestion Pipeline...")
    try:
        chunks = load_and_chunk_corpus(corpus_dir="corpus")
        print(f"\nSuccess! Loaded a total of {len(chunks)} chunks.")
        if chunks:
            print("\nExample Chunk Metadata:")
            print(chunks[0]["metadata"])
            print("\nExample Chunk Text (first 100 characters):")
            print(chunks[0]["text"][:100] + "...")
    except Exception as e:
        print(f"Ingestion testing failed: {e}")

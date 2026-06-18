# ASUS TUF Gaming F16 (2025) RAG Chatbot

An end-to-end Retrieval-Augmented Generation (RAG) chatbot designed in Python and Streamlit to answer user queries exclusively about the **ASUS TUF Gaming F16 (2025)** laptop. 

This project demonstrates the complete RAG pipeline: document ingestion, text chunking, local sentence-embedding generation, Pinecone index storage, LLM-based query intent classification, dynamic retrieval filtering, grounded LLM generation, and guardrail validation.

---

## Project Structure

```
asus_tuf_rag_project/
├── app.py                   # Streamlit UI & interactive frontend
├── ingest.py                # File reader, PDF extractor, text chunker, and metadata generator
├── vector_store.py          # Embedding generation & Pinecone integration (dimension 384, cosine metric)
├── intent_classifier.py     # Gemini Flash intent classification via OpenRouter
├── rag.py                   # Core orchestrator executing the retrieval and LLM generation
├── requirements.txt         # Required Python packages
├── README.md                # Documentation and setup instructions
└── corpus/                  # Local knowledge base
    ├── official_specs.txt   # Specs sheet for models, OS, CPU, RAM, etc.
    ├── user_manual.pdf      # Detailed ASUS manual PDF
    ├── faq.txt              # Standard Q&A on upgrades, connections, durability
    └── reviews.txt          # Combined list of positive and negative reviews
```

---

## System Architecture

The chatbot is built using the **Retrieval-Augmented Generation (RAG)** architecture. Below are the diagrams explaining the core concept, the ingestion infrastructure, and the end-user query workflows:

### 1. The Core RAG Concept (Retrieve -> Augment -> Generate)

Unlike standard LLMs that answer queries from static pre-trained weights, a RAG system first **retrieves** relevant document chunks from a custom database, **augments** the user's prompt with this real-time context, and then passes it to the LLM to **generate** a grounded response.

```mermaid
graph LR
    subgraph Retrieval [1. Retrieve]
        Query[User Query] -->|Semantic Search| VectorDB[(Pinecone Vector DB)]
        VectorDB -->|Top-K Chunks| Chunks[Relevant Text Passages]
    end

    subgraph Augmentation [2. Augment]
        Query & Chunks -->|Inject Context| ContextPrompt[Grounded LLM Prompt]
    end

    subgraph Generation [3. Generate]
        ContextPrompt -->|Sends Prompt| LLM[LLM / Generator]
        LLM -->|Grounded Answer| Answer[Final Answer]
    end
```

### 2. Ingestion & Indexing Pipeline (Infrastructure Build)

This pipeline reads raw text and PDF documents, segments them into overlap-protected chunks, extracts metadata tags, generates vector representations locally, and indexes them in Pinecone:

```mermaid
flowchart TD
    subgraph Data Prep [1. Knowledge Base Data Ingestion]
        Files[(corpus/)] -->|Reads txt/pdf| IngestEngine[ingest.py]
        IngestEngine -->|Loaders| Extract[Text Extraction / PdfReader]
    end

    subgraph Processing [2. Text Chunking & Metadata Tagging]
        Extract -->|Raw Text| Chunker[Chunking: ~500 words, 50-word overlap]
        Chunker -->|Chunk Blocks| Metadata[Add Metadata Tags:<br/>- source_file<br/>- document_type<br/>- chunk_id]
    end

    subgraph Embedding [3. Local Embedding Generation]
        Metadata -->|Text Chunks| LocalLLM[all-MiniLM-L6-v2 Model]
        LocalLLM -->|Generate Vector| Vectors[384-Dimensional Dense Vectors]
    end

    subgraph Indexing [4. Pinecone Vector Storage]
        Vectors -->|Batch Payload| VectorStore[vector_store.py]
        VectorStore -->|Verify Index| IndexCheck{Index exists?}
        IndexCheck -->|No| CreateIndex[Create Pinecone Index<br/>Metric: cosine]
        IndexCheck -->|Yes| ConnectIndex[Connect to Index]
        CreateIndex --> Upsert[Upsert Vectors & Metadata]
        ConnectIndex --> Upsert
        Upsert -->|Store in Namespace:<br/>asus_tuf_gaming_f16_2025| Pinecone[(Pinecone Index)]
    end
```

### 3. Real-time Query & Generation Pipeline (End-User Workflow)

This pipeline handles user interaction, categorizes user intent to apply dynamic filters, queries the Pinecone vector index, applies relevancy score guardrails, and constructs grounded responses using OpenRouter:

```mermaid
graph TD
    User([User]) -->|1. Enters Question| App[Streamlit UI - app.py]
    App -->|2. Executes Pipeline| Pipeline[RAG Orchestrator - rag.py]
    Pipeline -->|3. Classifies Intent| Classifier[Classifier - intent_classifier.py]
    Classifier -->|4. Sends Prompt| OpenRouter1[OpenRouter API]
    OpenRouter1 -->|5. Returns Intent| Classifier
    Classifier -->|6. Intent setup, spec, etc.| Pipeline
    Pipeline -->|7. Evaluates Category| Guard1{Is On-Topic?}
    Guard1 -->|No| Reject1[Return Default Grounded Rejection]
    Guard1 -->|Yes| Filter[Create Metadata Filter]
    Filter -->|8. Searches Namespace| Pinecone[(Pinecone DB)]
    Pinecone -->|9. Returns Chunks & Scores| Pipeline
    Pipeline -->|10. Evaluates Similarity Score| Guard2{Score >= 0.20?}
    Guard2 -->|No| Reject2[Return Default Grounded Rejection]
    Guard2 -->|Yes| Generation[Construct Context Prompt]
    Generation -->|11. Generates Response| OpenRouter2[OpenRouter API]
    OpenRouter2 -->|12. Grounded Answer| Pipeline
    Pipeline -->|13. Returns Answer & Trace| App
    App -->|14. Displays Answer to User| User
```

### Key Architectural Layers
1. **Frontend Layer (`app.py`)**: Built with Streamlit, incorporating modern CSS (dark theme, orange accent buttons, glassmorphic cards). It captures credentials, triggers manual document ingestion, and renders response blocks with internal pipeline diagnostics (retrieved chunk sources, score distributions, and intents).
2. **Orchestration Layer (`rag.py`)**: The central pipeline. It runs sequential stages, evaluating guardrails and fallbacks, and handles connection timeouts or Rate Limits (`429`) with automatic exponential backoffs.
3. **Intent Classification (`intent_classifier.py`)**: Categorizes questions beforehand to restrict the semantic search space to specific document types, filtering out background noise.
4. **Vector Database Layer (`vector_store.py`)**: Responsible for embedding text chunks utilizing `all-MiniLM-L6-v2` locally and performing high-performance cosine-similarity lookups in the Pinecone cloud.

---

## Project Screenshots

### 1. Main Application Dashboard
![Main Dashboard](./screenshots/app_main_page.png)

### 2. Live Document Ingestion & Pinecone Upload
![Document Ingestion](./screenshots/ingestion_running.png)

### 3. Pipeline Diagnostics and Retrieval Trace
![Pipeline Diagnostics](./screenshots/clicked_question.png)

---

## RAG Pipeline Stages Explained

### 1. Ingestion (`ingest.py`)
- **Extraction**: Supports both `.txt` (UTF-8) and `.pdf` (using `pypdf`'s `PdfReader`) inputs.
- **Chunking**: Uses a word-based splitter (~500 words per chunk with a 50-word overlap) to segment large documents, ensuring context isn't lost across chunk boundaries.
- **Metadata**: Saves custom tags for every chunk:
  - `source_file`: The file name (e.g. `official_specs.txt`).
  - `document_type`: Broad categories mapping to `spec`, `manual`, `faq`, or `review`.
  - `chunk_id`: Standard uniquely identified string.

### 2. Embeddings & Vector Store (`vector_store.py`)
- **Embeddings**: Employs the `sentence-transformers/all-MiniLM-L6-v2` model, which generates 384-dimensional dense vectors representing the semantic meaning of chunks.
- **Pinecone**: Connects to Pinecone. Creates an index named `asus-tuf-gaming-f16-2025` using `cosine` distance metric if it doesn't already exist.
- **Self-contained Storage**: The raw text of each chunk is uploaded directly in Pinecone's vector metadata under the key `"text"`. This simplifies retrieval by eliminating the need for a secondary document lookup database.

### 3. Intent Detection (`intent_classifier.py`)
- User queries are classified before search into one of six categories:
  - `specifications`, `setup/how_to`, `troubleshooting`, `reviews_opinions`, `general_product_question`, or `out_of_scope`.
- Intent detection is handled via OpenRouter with the `google/gemini-2.5-flash` model.
- **Retrieval Filter**: Chunks are selectively queried depending on the intent:
  - `specifications` -> queries are filtered to search only `spec` or `faq` chunks.
  - `setup/how_to` & `troubleshooting` -> filtered to search `manual` or `faq` chunks.
  - `reviews_opinions` -> filtered to search only `review` chunks.
  - `general_product_question` -> searches across all chunks without filter.

### 4. RAG Orchestrator & Grounded Generation (`rag.py`)
- **Guardrails**:
  - **Out of Scope Check**: If the intent is `out_of_scope`, the chatbot immediately returns `"I don't know from the available info."` without retrieving documents or calling the LLM.
  - **Similarity Threshold**: If the highest matching similarity score in Pinecone is below `0.20`, the chatbot rejects the query and outputs `"I don't know from the available info."` to prevent hallucinations on irrelevant topics.
- **Fallback Search**: If search with an intent filter yields no results or very low scores, the system automatically runs a fallback query without any metadata filter to avoid missing key details.
- **Prompt Engineering**: The final LLM prompt includes context documents and enforces a strict system instruction: answer *only* from context and output exact rejection text if the information is missing.

### 5. Frontend UI (`app.py`)
- A single-page Streamlit dashboard featuring custom dark/glassmorphic CSS.
- **Controls**: Includes a sidebar for API key configuration and a one-click button to execute ingestion.
- **Suggestions**: Quick questions to try out intent-based filtering and out-of-scope rejections.
- **Trace Diagnostic**: An expandable section detailing the internal pipeline execution data (intent, fallback status, retrieved chunks, metadata, and scores) for interview demonstrations.

---

## Installation and Execution

### Step 1: Install Dependencies
Create your virtual environment, activate it, and install dependencies:
```bash
pip install -r requirements.txt
```

### Step 2: Configure Environment Variables
Create a file named `.env` in the project root folder (or edit the template created) and add your keys:
```env
GEMINI_API_KEY=your_openrouter_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
```

### Step 3: Run the Streamlit Application
Start the frontend interface locally:
```bash
streamlit run app.py
```

### Step 4: Index the Knowledge Base
1. Open the Streamlit dashboard in your browser.
2. In the left sidebar, click the **"🚀 Run Ingestion & Indexing"** button.
3. This reads, chunks, generates embeddings, and indexes all 4 files into your Pinecone namespace `asus_tuf_gaming_f16_2025`.
4. Once completed, type any question in the input box and click **"🔍 Ask Chatbot"**.

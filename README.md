# AI Document Retrieval API (RAG)

A Retrieval-Augmented Generation (RAG) API built with FastAPI, LangChain, ChromaDB, HuggingFace Embeddings, and Groq LLM. It ingests PDF documents, creates vector store embeddings, and serves a query endpoint that returns grounded answers alongside source page citations.

---

## Features

- **PDF Ingestion**: Loads documents using `PyPDFLoader` and chunks text with `RecursiveCharacterTextSplitter`.
- **Local Vector Database**: Persists embeddings locally using ChromaDB (`chroma_db`).
- **Open Embeddings**: Uses HuggingFace `all-MiniLM-L6-v2` model for text embeddings.
- **Groq LLM Integration**: Fast inference with `ChatGroq` (`qwen/qwen3.6-27b`).
- **RESTful API**: Fast and light FastAPI endpoints for querying documents and checking server status.
- **Source Citations**: Returns exact source snippets and page metadata alongside every answer.

---

## Tech Stack

- **Framework**: FastAPI, Uvicorn
- **RAG & Orchestration**: LangChain, `langchain-community`, `langchain-chroma`, `langchain-groq`
- **Vector Store**: ChromaDB
- **Embeddings**: `sentence-transformers` / `huggingface-hub` (`all-MiniLM-L6-v2`)
- **Language Model**: Groq API

---

## Project Structure

```text
├── data/
│   └── sample.pdf          # Target PDF document for ingestion
├── chroma_db/              # Local vector database storage (generated)
├── ingest.py               # Document loading, chunking, and embedding script
├── rag_logic.py            # LangChain retrieval & QA chain logic
├── main.py                 # FastAPI application & REST endpoints
├── requirements.txt        # Python dependencies
└── .env                    # Environment variables (Groq API Key)
```

---

## Setup & Installation

### 1. Prerequisites
- Python 3.10+
- Groq API Key ([Get one here](https://console.groq.com/))

### 2. Clone & Environment Setup
```bash
git clone https://github.com/<your-username>/agentic-document-retrieval-api.git
cd agentic-document-retrieval-api

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Environment Variables
Create a `.env` file in the root directory:
```env
GROQ_API_KEY=your_groq_api_key_here
```

---

## Usage

### Step 1: Ingest Document
Place your PDF inside the `data/` directory as `sample.pdf` and run:

```bash
python ingest.py
```
This loads the PDF, splits it into chunks of 1000 characters (200 overlap), computes vector embeddings, and stores them in `chroma_db/`.

### Step 2: Test RAG Pipeline (CLI)
You can run the retrieval chain directly in CLI mode:

```bash
python rag_logic.py
```

### Step 3: Run the API Server
Start the Uvicorn server:

```bash
uvicorn main:app --reload
```
The API server will be available at `http://127.0.0.1:8000`.  
Swagger UI documentation is available at `http://127.0.0.1:8000/docs`.

---

## API Endpoints

### 1. `GET /health`
Returns the status of the API service.

**Response**:
```json
{
  "status": "healthy",
  "message": "api is running"
}
```

### 2. `POST /ask`
Submits a natural language query against the ingested document.

**Request Body**:
```json
{
  "query": "What is the main topic of this document?"
}
```

**Response Body**:
```json
{
  "answer": "The document is a systematic review that evaluates predictive models...",
  "sources": [
    {
      "page_content": "eligibility phase. For a rigorous evaluation...",
      "metadata": {
        "page": 1,
        "source": "data/sample.pdf"
      }
    }
  ]
}
```

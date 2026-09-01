import os
import shutil
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any

from rag_logic import get_answer
from ingest import ingest_pdf


app = FastAPI(
    title="Agentic RAG API",
    description="An API that performs retrieval-augmented generation on local PDFs.",
    version="1.0.0"
)

# --- CRUCIAL FIX: Enable CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows Streamlit Cloud to connect
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# --------------------------------

os.makedirs("data", exist_ok=True)

class QueryRequest(BaseModel):
    query:str
    history: List[Dict[str, str]] = []
class SourceCitation(BaseModel):
    page_content:str
    metadata:Dict[str,Any]
class QueryResponse(BaseModel):
    answer:str
    sources:List[SourceCitation]

@app.post("/ask",response_model=QueryResponse)
async def ask_question(request:QueryRequest):
    try:
        answer,docs=get_answer(request.query,request.history    )
        formatted_sources=[]
        for doc in docs:
            formatted_sources.append(
                SourceCitation(
                    page_content=doc.page_content,
                    metadata=doc.metadata
                )
            )
        return QueryResponse(
            answer=answer,
            sources=formatted_sources
        )
    except Exception as e:
        raise HTTPException(status_code=500,detail=str(e))

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        os.makedirs("data", exist_ok=True)
        file_path = f"data/{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        
        ingest_pdf(file_path)
        
        return {"filename": file.filename, "message": "File successfully uploaded and vectorized!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Agentic RAG API is running. Visit /docs for the API documentation."}

@app.get("/health")
async def health_check():
    return {"status":"healthy","message":"api is running"}



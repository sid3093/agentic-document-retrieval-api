from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any

from rag_logic import get_answer

app = FastAPI(
    title="Agentic RAG API",
    description="An API that performs retrieval-augmented generation on local PDFs.",
    version="1.0.0"
)

class QueryRequest(BaseModel):
    query:str
class SourceCitation(BaseModel):
    page_content:str
    metadata:Dict[str,Any]
class QueryResponse(BaseModel):
    answer:str
    sources:List[SourceCitation]

@app.post("/ask",response_model=QueryResponse)
async def ask_question(request:QueryRequest):
    try:
        answer,docs=get_answer(request.query)
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
@app.get("/health")
async def health_check():
    return {"status":"healthy","message":"api is running"}
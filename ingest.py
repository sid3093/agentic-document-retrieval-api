import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

PDF_PATH="data/sample.pdf"
CHROMA_PATH="chroma_db"

def ingest_document():
    print(f"Loading PDF from {PDF_PATH}...")
    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(f"Please put a PDF in the path")
    loader=PyPDFLoader(PDF_PATH)
    documents=loader.load()
    print("splitting into chunksss")
    text_splitter=RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks=text_splitter.split_documents(documents)
    print(f"Split document into {len(chunks)} chunks.")
    print("Initialising embedding model")
    embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    print("Saving chunks to ChromaDB")
    vector_store=Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )
    print("Data ingestion complete")



if __name__=="__main__":
    ingest_document()

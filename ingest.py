import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

PDF_PATH="data/sample.pdf"
CHROMA_PATH="chroma_db"

def ingest_document(file_path: str = PDF_PATH):
    print(f"Loading PDF from {file_path}...")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Please put a PDF in the path: {file_path}")
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    print("Splitting into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split document into {len(chunks)} chunks.")
    print("Initializing embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    print("Clearing old vector database...")
    old_db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    try:
        old_db.delete_collection() # This deletes all existing vectors
    except Exception:
        pass
    print("Saving chunks to ChromaDB...")
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )
    print("Data ingestion complete")
    return True

ingest_pdf = ingest_document


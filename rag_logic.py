import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
import re
load_dotenv()
CHROMA_PATH="chroma_db"
embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store=Chroma(
    persist_directory=CHROMA_PATH,
    embedding_function=embeddings
)
retriever=vector_store.as_retriever(search_kwargs={"k":3})
llm=ChatGroq(
    temperature=0,
    model_name="qwen/qwen3.6-27b"
)
system_prompt=(
    "You are an intelligent AI assistant. Use the following pieces of retrieved "
    "context to answer the question. If you don't know the answer based on the "
    "context, just say that you don't know. Keep the answer concise and clear.\n\n"
    "Context: {context}"
)
prompt=ChatPromptTemplate.from_messages([
    ("system",system_prompt),
    ("human","{input}"),
])
question_answer_chain=create_stuff_documents_chain(llm,prompt)
rag_chain=create_retrieval_chain(retriever,question_answer_chain)
def get_answer(query: str):
    print(f"\nThinking about: '{query}'...")
    response = rag_chain.invoke({"input": query})
    
    answer = response["answer"]
    
    answer = re.sub(r'<think>.*?</think>\n*', '', answer, flags=re.DOTALL).strip()
    
    sources = response["context"]
    return answer, sources
if __name__=="__main__":
    test_query="what is this document about?give me a brief summary"
    ans,docs=get_answer(test_query)
    print("\n==ANSWER==")
    print(ans)
    print("\n==SOURCES==")
    for i,doc in enumerate(docs):
        print(f"Source {i+1}: (Page {doc.metadata.get('page', 'Unknown')}) {doc.page_content[:150]}...")

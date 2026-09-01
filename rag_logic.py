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
retriever=vector_store.as_retriever(search_kwargs={"k":4})
llm=ChatGroq(
    temperature=0.3,
    model_name="qwen/qwen3-32b",
    reasoning_format="hidden"
)
system_prompt=(
    "You are an expert AI research assistant. Use the provided context to answer the user's question thoroughly.\n"
    "If you don't know the answer based on the context, just say that you don't know.\n\n"
    "Context: {context}"
)
prompt=ChatPromptTemplate.from_messages([
    ("system",system_prompt),
    ("human","Conversation History:\n{history}\n\nNew Question: {input}"),
])
question_answer_chain=create_stuff_documents_chain(llm,prompt)
rag_chain=create_retrieval_chain(retriever,question_answer_chain)
def get_answer(query: str, history: list = []):
    history_string = ""
    for msg in history:
        role = "User" if msg["role"] == "user" else "AI"
        history_string += f"{role}: {msg['content']}\n"

    
    if not history_string:
        history_string = "No previous history."

    # Invoke the chain, passing in the input AND the history
    response = rag_chain.invoke({
        "input": query,
        "history": history_string
    })
    
    raw_answer = response["answer"]
    clean_answer = re.sub(r"<think>.*?</think>", "", raw_answer, flags=re.DOTALL).strip()
    
    return clean_answer, response["context"]
if __name__=="__main__":
    test_query="what is this document about?give me a brief summary"
    ans,docs=get_answer(test_query)
    print("\n==ANSWER==")
    print(ans)
    print("\n==SOURCES==")
    for i,doc in enumerate(docs):
        print(f"Source {i+1}: (Page {doc.metadata.get('page', 'Unknown')}) {doc.page_content[:150]}...")

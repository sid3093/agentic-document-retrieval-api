import os
import re
import tempfile
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

st.set_page_config(page_title="Agentic PDF Assistant", page_icon="📄", layout="wide")
st.title("📄 Agentic PDF Assistant")

# Retrieve Groq API Key from environment or Streamlit Secrets
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key and "GROQ_API_KEY" in st.secrets:
    groq_api_key = st.secrets["GROQ_API_KEY"]

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "uploaded_filename" not in st.session_state:
    st.session_state.uploaded_filename = None

# Cache the Embedding Model to save RAM
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

embeddings = load_embeddings()

# Sidebar: Document Upload
with st.sidebar:
    st.header("1. Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF", type="pdf")
    
    if uploaded_file is not None and st.session_state.uploaded_filename != uploaded_file.name:
        with st.spinner(f"Vectorizing {uploaded_file.name}..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name

            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
            
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = splitter.split_documents(docs)

            # Create in-memory/session-scoped vector store
            st.session_state.vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings
            )
            
            os.remove(tmp_path)
            st.session_state.uploaded_filename = uploaded_file.name
            st.session_state.messages = []  # Clear chat history for fresh document
            st.success(f"{uploaded_file.name} is ready!")

    st.divider()
    st.markdown("### Source Citations")

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Query Processing
# Replace your existing chain.invoke block with this:

if prompt := st.chat_input("Ask a question about your PDF..."):
    if not st.session_state.vector_store:
        st.warning("Please upload a PDF document first.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing document..."):
            try:
                retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 5})
                groq_api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
                if not groq_api_key:
                    st.error("CRITICAL: GROQ_API_KEY is missing from Streamlit Secrets!")
                    st.stop()

                llm = ChatGroq(
                    model_name="llama-3.3-70b-versatile",
                    temperature=0.3,
                    api_key=groq_api_key
                )

                system_prompt = (
                    "You are an expert AI research assistant. Use the provided context to answer the user's question thoroughly.\n"
                    "If you don't know the answer based on the context, just say that you don't know.\n\n"
                    "Context: {context}"
                )

                prompt_template = ChatPromptTemplate.from_messages([
                    ("system", system_prompt),
                    ("human", "Conversation History:\n{history}\n\nNew Question: {input}"),
                ])

                history_str = ""
                for msg in st.session_state.messages[:-1]:
                    role = "User" if msg["role"] == "user" else "AI"
                    history_str += f"{role}: {msg['content']}\n"

                chain = create_retrieval_chain(retriever, create_stuff_documents_chain(llm, prompt_template))
                response = chain.invoke({"input": prompt, "history": history_str or "No previous history."})

                raw_answer = response["answer"]
                clean_answer = re.sub(r"<think>.*?</think>", "", raw_answer, flags=re.DOTALL).strip()

                st.markdown(clean_answer)
                st.session_state.messages.append({"role": "assistant", "content": clean_answer})

                if "context" in response:
                    with st.sidebar:
                        st.divider()
                        st.markdown("**Sources for latest query:**")
                        for i, doc in enumerate(response["context"]):
                            page = doc.metadata.get("page", "Unknown")
                            with st.expander(f"Source {i+1} (Page {page})"):
                                st.write(doc.page_content)
                                
            except Exception as e:
                # THIS WILL PRINT THE EXACT GROQ ERROR TO YOUR SCREEN IN RED
                st.error(f"DETAILED ERROR: {str(e)}")
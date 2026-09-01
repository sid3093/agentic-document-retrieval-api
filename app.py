import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Agentic PDF Chat", page_icon="📄", layout="wide")
st.title("📄 Agentic PDF Assistant")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- SIDEBAR: AUTO-FILE UPLOAD ---
with st.sidebar:
    st.header("1. Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF", type="pdf")
    
    # AUTO-PROCESS LOGIC: Trigger immediately when a file is uploaded
    if uploaded_file is not None:
        # Check if we already uploaded THIS exact file to avoid infinite reloading loops
        if "uploaded_filename" not in st.session_state or st.session_state.uploaded_filename != uploaded_file.name:
            with st.spinner(f"Vectorizing {uploaded_file.name}..."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                response = requests.post(f"{API_URL}/upload", files=files)
                
                if response.status_code == 200:
                    st.success(f"{uploaded_file.name} is ready!")
                    st.session_state.uploaded_filename = uploaded_file.name
                    # Clear chat history for the new document
                    st.session_state.messages = []
                else:
                    st.error(f"Error: {response.text}")

    st.divider()
    st.markdown("### Source Citations")

# --- MAIN CHAT UI ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about your PDF..."):
    # Add user message to UI
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # NEW: We are now sending the entire chat history in the JSON payload!
                payload = {
                    "query": prompt,
                    "history": st.session_state.messages[:-1] # Send all messages EXCEPT the current one
                }
                
                response = requests.post(f"{API_URL}/ask", json=payload)
                
                if response.status_code == 200:
                    data = response.json()
                    answer = data.get("answer")
                    sources = data.get("sources", [])
                    
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    
                    if sources:
                        with st.sidebar:
                            st.divider()
                            st.markdown(f"**Sources for: '{prompt}'**")
                            for i, source in enumerate(sources):
                                page = source.get("metadata", {}).get("page", "Unknown")
                                with st.expander(f"Source {i+1} (Page {page})"):
                                    st.write(source.get("page_content"))
                else:
                    st.error(f"API Error: {response.status_code}")
                    
            except requests.exceptions.ConnectionError:
                st.error("Failed to connect to the backend. Is FastAPI running?")
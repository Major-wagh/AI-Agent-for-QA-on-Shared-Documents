import streamlit as st
import os
from src.qa_agent import QAAgent
from src.embeddings import create_and_store_embeddings
from dotenv import load_dotenv

load_dotenv()

# Initialize the QA Agent
@st.cache_resource
def get_qa_agent():
    """Caches the QA Agent to avoid re-initializing on every rerun."""
    try:
        agent = QAAgent()
        if agent.vector_store is None:
            st.error("Failed to load FAISS vector store. Please ensure 'faiss_excel_index' exists and is valid.")
            return None
        return agent
    except ValueError as e:
        st.error(f"Configuration Error: {e}. Please set your GOOGLE_API_KEY in the .env file.")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred during QA Agent initialization: {e}")
        return None

qa_agent = get_qa_agent()

st.set_page_config(page_title="AI Agent for QA on Shared Documents", layout="wide")

st.title("📄 AI Agent for QA on Shared Documents")
st.markdown("Ask questions about your Excel documents and get answers powered by AI.")

# Sidebar for information and actions
with st.sidebar:
    st.header("About This Application")
    st.info(
        "This AI agent answers questions based on pre-processed Excel documents. "
        "The embeddings for `data.xlsx` and `Forcast.xlsx` are already created "
        "and stored in the `faiss_excel_index` directory."
    )
    
    st.header("Document Status")
    if qa_agent and qa_agent.vector_store:
        st.success("FAISS Vector Store Loaded Successfully!")
        st.write("Ready to answer questions based on the pre-indexed documents.")
    else:
        st.warning("FAISS Vector Store Not Loaded.")
        st.write("Please ensure the `faiss_excel_index` directory exists and contains valid FAISS index files.")
    
    st.markdown("---")
    st.subheader("Upload New Documents")
    uploaded_files = st.file_uploader(
        "Upload documents (Excel)", 
        type=["xlsx"], 
        accept_multiple_files=True
    )

    if uploaded_files:
        temp_dir = "uploaded_temp"
        os.makedirs(temp_dir, exist_ok=True)
        uploaded_file_paths = []
        for uploaded_file in uploaded_files:
            file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            uploaded_file_paths.append(file_path)
        
        if st.button("Process Uploaded Documents"):
            with st.spinner("Processing and embedding uploaded documents... This may take a while."):
                # Combine existing files with uploaded files for re-embedding
                existing_files = ["./data/data.xlsx", "./data/Forcast.xlsx"]
                all_files_to_embed = list(set(existing_files + uploaded_file_paths)) # Use set to avoid duplicates
                
                # Clear existing cache to ensure new embeddings are generated
                # This is a simplified approach; a more robust solution would merge indices
                # For this assignment, re-creating the entire index is acceptable.
                from src.embeddings import DocumentEmbeddings
                processor = DocumentEmbeddings()
                processor.clear_cache() # Clear cache before re-embedding all files
                
                vector_store = create_and_store_embeddings(all_files_to_embed)
                if vector_store:
                    st.success("Documents processed, embeddings created, and FAISS index updated!")
                    st.rerun() # Rerun to reload the agent with updated index
                else:
                    st.error("Failed to process uploaded documents. Check console for details.")
        
    st.markdown("---")
    st.subheader("Re-create All Embeddings (if needed)")
    st.write("This will re-process all default documents and clear any uploaded ones.")
    if st.button("Re-create Default Embeddings"):
        with st.spinner("Creating embeddings for default documents... This may take a while."):
            from src.embeddings import DocumentEmbeddings
            processor = DocumentEmbeddings()
            processor.clear_cache() # Clear cache before re-embedding default files
            
            vector_store = create_and_store_embeddings(["./data/data.xlsx", "./data/Forcast.xlsx"])
            if vector_store:
                st.success("Default embeddings created and FAISS index saved!")
                st.rerun() # Rerun to reload the agent
            else:
                st.error("Failed to create default embeddings. Check console for details.")

# Main content area
st.header("Ask a Question")
user_question = st.text_area("Enter your question here:", height=100)

if st.button("Get Answer"):
    if qa_agent:
        if user_question:
            with st.spinner("Searching for answer..."):
                answer = qa_agent.ask_question(user_question)
                st.subheader("Answer:")
                st.write(answer)
        else:
            st.warning("Please enter a question.")
    else:
        st.error("QA Agent is not initialized. Cannot process your question.")

st.markdown("---")
st.caption("Developed as an AI Agent for QA on Shared Documents.")

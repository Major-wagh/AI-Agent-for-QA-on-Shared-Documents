# AI Agent for QA on Shared Documents - Comprehensive Documentation

## 1. Overview

This document provides comprehensive documentation for the AI-powered agent designed to answer questions based on shared Excel, PDF, and Word documents. The project integrates advanced Natural Language Processing (NLP) techniques with a user-friendly Streamlit interface, enabling efficient user interaction and information retrieval.

The core functionality revolves around leveraging Google's Generative AI models for creating document embeddings and performing question-answering. A FAISS vector store is utilized for efficient similarity search over the pre-processed and embedded document content. The Streamlit UI serves as the interactive front-end, allowing users to upload new documents, manage existing embeddings, input questions, and view AI-generated answers.

## 2. Design Decisions and Architecture

The architecture of this AI agent is modular, designed for scalability, maintainability, and performance. It is divided into three primary components: Embedding Generation, Question Answering (QA) Agent, and User Interface (UI).

### 2.1. System Architecture Diagram

```mermaid
graph TD
    User[User] -->|Asks Question / Uploads Docs| StreamlitApp(Streamlit UI - app.py)

    subgraph QA Process
        StreamlitApp -->|Query| QAAgent(QA Agent - src/qa_agent.py)
        QAAgent -->|Similarity Search| FAISSStore(FAISS Vector Store)
        FAISSStore -->|Relevant Chunks| QAAgent
        QAAgent -->|Chunks + Query| LLM(Google Generative AI Model)
        LLM -->|Answer| QAAgent
        QAAgent -->|Answer| StreamlitApp
    end

    subgraph Document Processing & Embedding
        Documents[Excel, PDF, Word Documents (data/, uploaded_temp/)] -->|Load & Process| EmbeddingGen(Embedding Generation - src/embeddings.py)
        EmbeddingGen -->|Embeddings| FAISSStore
        StreamlitApp -->|Trigger Re-embed / Upload| EmbeddingGen
    end
```

### 2.2. Component Breakdown and Design Decisions

#### 2.2.1. Embedding Generation (`src/embeddings.py`)

This component is responsible for transforming raw document data (Excel, PDF, Word) into a format suitable for semantic search.

*   **Technology Choice: Polars for Excel, PyPDFLoader for PDF, Docx2txtLoader for Word**:
    *   **Decision**: Polars is used for Excel due to its performance. `PyPDFLoader` and `Docx2txtLoader` from `langchain_community.document_loaders` are used for PDF and Word documents, respectively.
    *   **Reasoning**: This provides robust support for multiple document types, fulfilling the requirement for diverse document uploads. Each loader is specialized for its format, ensuring accurate text extraction.
*   **Streaming Processing for Large Excel Files**:
    *   **Decision**: Implemented `process_large_excel_streaming` to handle very large Excel files in chunks.
    *   **Reasoning**: Processing entire large files in memory can lead to out-of-memory errors. Streaming allows the application to process data incrementally, managing memory usage effectively and enabling the handling of files that would otherwise be too large. This is specifically applied to Excel files where Polars can leverage this.
*   **Caching Mechanism**:
    *   **Decision**: A caching mechanism (`process_file_with_cache`) is implemented to store processed document chunks.
    *   **Reasoning**: Re-processing and re-embedding unchanged documents is computationally expensive and time-consuming. Caching ensures that if a file has not been modified, its pre-computed chunks are loaded directly, speeding up subsequent runs and reducing API calls. File hashing based on size and modification time is used for efficient cache invalidation.
*   **Parallel Embedding Creation**:
    *   **Decision**: `create_embeddings_parallel` utilizes `ThreadPoolExecutor` for generating embeddings in parallel.
    *   **Reasoning**: Embedding generation involves API calls (I/O-bound operations). Parallel processing allows multiple batches of embeddings to be generated concurrently, significantly reducing the overall time required, while small delays are introduced to manage API rate limits.
*   **Document Representation**:
    *   **Decision**: All document types are converted into `langchain.schema.Document` objects.
    *   **Reasoning**: This standard format allows seamless integration with LangChain's text splitting and vector store functionalities. Metadata (source file, sheet name/page number, dimensions, file type) is preserved for better context during retrieval.
*   **Text Splitting**:
    *   **Decision**: `RecursiveCharacterTextSplitter` is used for chunking documents.
    *   **Reasoning**: This splitter is effective for maintaining semantic coherence within chunks by attempting to split on various characters in a hierarchical order, which is crucial for accurate retrieval.

#### 2.2.2. Question Answering (QA) Agent (`src/qa_agent.py`)

This component forms the brain of the application, responsible for understanding questions and generating answers.

*   **Technology Choice: LangGraph ReAct Agent**:
    *   **Decision**: Switched from a simple `load_qa_chain` to `langgraph.prebuilt.create_react_agent`.
    *   **Reasoning**: The ReAct (Reasoning and Acting) agent architecture provides a more robust and flexible approach to question answering. It allows the LLM to dynamically decide whether to use tools (like a retriever) to gather information before formulating an answer. This enhances the agent's ability to handle complex queries, reason over retrieved documents, and potentially perform multi-step reasoning. It moves beyond simple retrieval-augmented generation to a more intelligent, tool-using agent.
*   **LLM Initialization**:
    *   **Decision**: `ChatGoogleGenerativeAI(model="gemini-2.0-flash-001")` is used.
    *   **Reasoning**: `gemini-2.0-flash-001` is a powerful and efficient model suitable for conversational AI and question-answering tasks. The `temperature=0.3` setting promotes more focused and less creative answers, which is desirable for factual QA based on documents.
*   **Retriever Tool Integration**:
    *   **Decision**: `create_retriever_tool` is used to expose the FAISS vector store as a tool for the ReAct agent.
    *   **Reasoning**: This allows the agent to explicitly "search" the document embeddings when it determines that external knowledge is required to answer a user's question. The tool is given a descriptive name and description (`excel_document_retriever`, "Searches and returns information from Excel documents.") to guide the LLM's tool-use reasoning.
*   **FAISS Vector Store**:
    *   **Decision**: FAISS (Facebook AI Similarity Search) is used for storing and searching embeddings.
    *   **Reasoning**: FAISS is highly optimized for similarity search on large datasets of vectors, providing fast and efficient retrieval of relevant document chunks. `allow_dangerous_deserialization=True` is used for loading the pickled FAISS index, acknowledging the security implications but necessary for this local setup.

#### 2.2.3. User Interface (UI) (`app.py`)

This component provides the interactive front-end for the application.

*   **Technology Choice: Streamlit**:
    *   **Decision**: Streamlit was chosen for building the UI.
    *   **Reasoning**: Streamlit allows for rapid development of interactive web applications using pure Python. Its simplicity and component-based structure make it ideal for quickly prototyping and deploying data-centric applications like this QA agent.
*   **Caching QA Agent**:
    *   **Decision**: The `get_qa_agent()` function is decorated with `@st.cache_resource`.
    *   **Reasoning**: Streamlit reruns the entire script on every user interaction. Caching the `QAAgent` instance prevents re-initializing the LLM and re-loading the large FAISS vector store on every rerun, significantly improving application responsiveness and performance.
*   **Document Upload Functionality**:
    *   **Decision**: Implemented `st.file_uploader` to allow users to upload new Excel, PDF, and Word documents.
    *   **Reasoning**: This fulfills a key requirement for users to interact with their own shared documents. Uploaded files are temporarily stored, processed, and then used to update the FAISS index. The `type` parameter in `st.file_uploader` is set to `["xlsx", "pdf", "docx"]` to accept these formats.
*   **Dynamic Embedding Recreation**:
    *   **Decision**: Added buttons to "Process Uploaded Documents" and "Re-create Default Embeddings".
    *   **Reasoning**: This provides flexibility. Users can either augment the existing knowledge base with new documents or completely reset and re-index the default documents. Clearing the cache (`processor.clear_cache()`) before re-embedding ensures that the new embeddings are generated from scratch, preventing stale data issues. `st.experimental_rerun()` is used to force Streamlit to reload the application and pick up the newly created FAISS index.
*   **User Feedback**:
    *   **Decision**: Utilized Streamlit's built-in status messages (`st.success`, `st.warning`, `st.error`, `st.spinner`).
    *   **Reasoning**: Providing clear visual feedback to the user about the status of operations (e.g., loading, processing, errors) enhances the user experience and helps in troubleshooting.


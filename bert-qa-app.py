import streamlit as st
import torch
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders import JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import LlamaCpp
import os
import tempfile
import logging
from typing import List, Optional
from pathlib import Path
from langchain_core.documents import Document
import requests
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Function to download the model
def download_model(url: str, local_path: Path) -> None:
    """
    Download the LLM model file if it doesn't exist.
    """
    if not local_path.exists():
        st.info("Downloading model file (this may take a few minutes)...")
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(local_path, 'wb') as file, tqdm(
            desc=local_path.name,
            total=total_size,
            unit='iB',
            unit_scale=True
        ) as progress_bar:
            for data in response.iter_content(chunk_size=1024):
                size = file.write(data)
                progress_bar.update(size)

# Cache the models loading to improve performance
@st.cache_resource
def load_models():
    """
    Load and cache both BERT and LLama models.
    Returns:
        tuple: (SentenceTransformer, LlamaCpp)
    """
    try:
        # Load BERT model
        bert_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        
        # Set up model paths
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        model_path = models_dir / "llama-2-7b-chat.Q4_K_M.gguf"
        
        # Download model if needed
        if not model_path.exists():
            model_url = "https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf"
            download_model(model_url, model_path)
        
        # Initialize LlamaCpp
        llm = LlamaCpp(
            model_path=str(model_path),
            temperature=0.75,
            max_tokens=2000,
            top_p=1,
            n_ctx=2048,
            verbose=False
        )
        
        return bert_model, llm
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        raise RuntimeError(f"Failed to load models: {str(e)}")

def save_uploaded_file(uploaded_file) -> str:
    """
    Save the uploaded file to a temporary location.
    Args:
        uploaded_file: StreamlitUploadedFile object
    Returns:
        str: Path to the saved file
    """
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            return tmp_file.name
    except Exception as e:
        logger.error(f"Error saving uploaded file: {e}")
        raise RuntimeError(f"Failed to save uploaded file: {str(e)}")

def load_documents(file_path: str, file_type: str) -> Optional[List[Document]]:
    """
    Load documents from a PDF or JSON file.
    Args:
        file_path: Path to the file
        file_type: MIME type of the file
    Returns:
        List[Document]: Loaded documents or None if loading fails
    """
    try:
        if file_type == "application/pdf":
            loader = PyMuPDFLoader(file_path)
        elif file_type == "application/json":
            loader = JSONLoader(
                file_path,
                jq_schema=".",
                text_content=False,
                json_lines=False
            )
        else:
            raise ValueError("Unsupported file type")
        
        documents = loader.load()
        logger.info(f"Successfully loaded {len(documents)} documents")
        return documents
    except Exception as e:
        logger.error(f"Error loading documents: {e}")
        return None

def split_documents(documents: List[Document], chunk_size: int = 500, chunk_overlap: int = 50) -> List[Document]:
    """
    Split documents into smaller chunks for processing.
    Args:
        documents: List of documents to split
        chunk_size: Size of each chunk
        chunk_overlap: Overlap between chunks
    Returns:
        List[Document]: Split documents
    """
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False
        )
        split_docs = text_splitter.split_documents(documents)
        logger.info(f"Split documents into {len(split_docs)} chunks")
        return split_docs
    except Exception as e:
        logger.error(f"Error splitting documents: {e}")
        raise RuntimeError(f"Failed to split documents: {str(e)}")

def build_vector_store(documents: List[Document], model: SentenceTransformer) -> FAISS:
    """
    Build a FAISS vector store from documents.
    Args:
        documents: List of documents to index
        model: SentenceTransformer model for embeddings
    Returns:
        FAISS: Vector store
    """
    try:
        texts = [doc.page_content for doc in documents]
        embeddings = model.encode(texts, convert_to_tensor=True, show_progress_bar=True)
        embeddings = embeddings.cpu().numpy()  # Convert to numpy array
        
        faiss_store = FAISS.from_embeddings(
            text_embeddings=list(zip(texts, embeddings)),
            embedding=model,
            metadatas=[doc.metadata for doc in documents]
        )
        return faiss_store
    except Exception as e:
        logger.error(f"Error building vector store: {e}")
        raise RuntimeError(f"Failed to build vector store: {str(e)}")

def format_context_and_question(context: str, question: str) -> str:
    """
    Format the context and question for the LLM.
    """
    return f"""Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}

Question: {question}

Answer: """

def get_qa_response(llm: LlamaCpp, vector_store: FAISS, query: str) -> str:
    """
    Get response from the QA system.
    """
    # Get relevant documents from vector store
    docs = vector_store.similarity_search(query, k=3)
    context = "\n\n".join(doc.page_content for doc in docs)
    
    # Format prompt and get response
    prompt = format_context_and_question(context, query)
    response = llm(prompt)
    
    return response

def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Document Q&A System",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("📚 Document Q&A System")
    st.markdown("""
    Upload a PDF or JSON file and ask questions about its contents.
    The system uses BERT embeddings and a local LLM for document analysis.
    """)

    # Initialize session state
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None
        st.session_state.llm = None

    # Load models at startup
    try:
        bert_model, llm = load_models()
        st.session_state.llm = llm
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return

    # File upload section
    uploaded_file = st.file_uploader(
        "Upload a PDF or JSON file",
        type=["pdf", "json"],
        help="Select a PDF or JSON file to analyze"
    )

    if uploaded_file is not None:
        try:
            with st.spinner("Processing document..."):
                # Save uploaded file
                file_path = save_uploaded_file(uploaded_file)
                
                # Load and process documents
                documents = load_documents(file_path, uploaded_file.type)
                if documents is None:
                    st.error("Failed to load document. Please check the file format.")
                    return

                # Process documents
                split_docs = split_documents(documents)
                st.session_state.vector_store = build_vector_store(split_docs, bert_model)
                
                st.success("✅ Document processed successfully!")
                
                # Clean up temporary file
                os.unlink(file_path)

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            logger.exception("Error in document processing")
            return

    # Question input section
    if st.session_state.vector_store is not None and st.session_state.llm is not None:
        query = st.text_input(
            "Ask a question about the document:",
            placeholder="Enter your question here..."
        )

        if query:
            try:
                with st.spinner("Thinking..."):
                    response = get_qa_response(st.session_state.llm, st.session_state.vector_store, query)
                    st.write("Answer:", response)
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")
                logger.exception("Error in question answering")

if __name__ == "__main__":
    main()

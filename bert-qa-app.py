import streamlit as st
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch
from urllib.parse import urljoin
import PyPDF2
import io
import fitz  # PyMuPDF
import tempfile
import os

# Initialize BERT models
@st.cache_resource
def load_models():
    try:
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        qa_model = pipeline('question-answering', model='deepset/minilm-uncased-squad2')
        return embedding_model, qa_model
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None

def extract_pdf_content(pdf_file):
    try:
        content = ""
        extracted_text = []
        
        with st.expander("📄 Extracted PDF Content", expanded=False):
            status_text = st.empty()
            
            # Create a temporary file to store the PDF
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                temp_file.write(pdf_file.getvalue())
                temp_path = temp_file.name

            try:
                # Use PyMuPDF (fitz) for better text extraction
                doc = fitz.open(temp_path)
                total_pages = doc.page_count
                
                progress_bar = st.progress(0)
                
                for page_num in range(total_pages):
                    status_text.text(f"Processing page {page_num + 1}/{total_pages}")
                    
                    page = doc[page_num]
                    page_text = page.get_text()
                    
                    # Clean the text
                    page_text = ' '.join(page_text.split())
                    
                    content += page_text + "\n\n"
                    extracted_text.append(f"Page {page_num + 1}:\n{page_text[:500]}...\n\n")
                    
                    progress_bar.progress((page_num + 1) / total_pages)
                
                doc.close()
                
                status_text.text("PDF processing completed!")
                st.text_area("Extracted Content Preview", '\n'.join(extracted_text), height=200)
                
            finally:
                # Clean up the temporary file
                os.unlink(temp_path)
                
        return content
        
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        return None

def fetch_website_content(url, max_pages=5):
    try:
        visited = set()
        to_visit = [url]
        content = ""
        extracted_text = []
        
        with st.expander("🌐 Extracted Website Content", expanded=False):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i in range(max_pages):
                if not to_visit:
                    break
                    
                current_url = to_visit.pop(0)
                if current_url in visited:
                    continue
                
                status_text.text(f"Fetching page {i+1}/{max_pages}: {current_url}")
                
                try:
                    headers = {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                    }
                    response = requests.get(current_url, headers=headers, timeout=10)
                    response.raise_for_status()
                    
                    soup = BeautifulSoup(response.content, 'html.parser')
                    
                    # Remove script and style elements
                    for script in soup(["script", "style"]):
                        script.decompose()
                    
                    # Get text and clean it
                    page_text = soup.get_text(separator=' ', strip=True)
                    page_text = ' '.join(page_text.split())
                    
                    content += page_text + "\n\n"
                    extracted_text.append(f"Page {i+1}: {current_url}\n{page_text[:500]}...\n\n")
                    
                    # Find links
                    links = soup.find_all('a', href=True)
                    for link in links:
                        href = link['href']
                        full_url = urljoin(current_url, href)
                        if full_url.startswith(url) and full_url not in visited:
                            to_visit.append(full_url)
                    
                    visited.add(current_url)
                    
                except requests.RequestException as e:
                    st.warning(f"Error fetching {current_url}: {str(e)}")
                    continue
                
                progress_bar.progress((i + 1) / max_pages)
                
            status_text.text("Content extraction completed!")
            st.text_area("Extracted Content Preview", '\n'.join(extracted_text), height=200)
        
        return content
        
    except Exception as e:
        st.error(f"Error during content extraction: {str(e)}")
        return None

def split_text(text, chunk_size=500):
    if not text:
        return []
    
    sentences = text.replace('\n', ' ').split('. ')
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        if current_length + len(sentence) > chunk_size:
            chunks.append('. '.join(current_chunk) + '.')
            current_chunk = [sentence]
            current_length = len(sentence)
        else:
            current_chunk.append(sentence)
            current_length += len(sentence)
    
    if current_chunk:
        chunks.append('. '.join(current_chunk) + '.')
    
    return chunks

def create_embeddings(chunks, embedding_model):
    try:
        with st.spinner("Creating embeddings..."):
            return embedding_model.encode(chunks)
    except Exception as e:
        st.error(f"Error creating embeddings: {str(e)}")
        return None

def find_most_relevant_chunk(question, chunks, chunk_embeddings, embedding_model):
    try:
        question_embedding = embedding_model.encode([question])[0]
        similarities = cosine_similarity([question_embedding], chunk_embeddings)[0]
        most_similar_idx = np.argmax(similarities)
        
        # Show relevant context
        with st.expander("📚 Relevant Context", expanded=False):
            st.write("The answer was derived from this context:")
            st.write(chunks[most_similar_idx])
            
        return chunks[most_similar_idx]
    except Exception as e:
        st.error(f"Error finding relevant content: {str(e)}")
        return None

def answer_question(question, context, qa_pipeline):
    try:
        if not context:
            return "Sorry, I couldn't find relevant content to answer your question."
            
        result = qa_pipeline(question=question, context=context)
        return result['answer']
    except Exception as e:
        st.error(f"Error generating answer: {str(e)}")
        return "Sorry, I encountered an error while trying to answer your question."

# Streamlit UI
st.set_page_config(page_title="Document Q&A Chatbot (BERT)", layout="wide")

st.title("Document Q&A Chatbot (BERT)")
st.markdown("""
    <style>
        .stTextInput > div > div > input {
            background-color: #1E1E1E;
            color: white;
        }
    </style>
    """, unsafe_allow_html=True)

# Load models
with st.spinner("Loading models..."):
    embedding_model, qa_pipeline = load_models()

if embedding_model is None or qa_pipeline is None:
    st.error("Failed to load models. Please refresh the page.")
    st.stop()

# Session state for tracking processing status
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'content' not in st.session_state:
    st.session_state.content = None
if 'chunks' not in st.session_state:
    st.session_state.chunks = None
if 'chunk_embeddings' not in st.session_state:
    st.session_state.chunk_embeddings = None

# Input method selection
input_method = st.radio("Select input method:", ["Website URL", "PDF File"])

if input_method == "Website URL":
    url = st.text_input("Enter website URL:", help="Enter the full URL including http:// or https://")
    if url:
        if not (url.startswith('http://') or url.startswith('https://')):
            st.error("Please enter a valid URL starting with http:// or https://")
            st.stop()
        
        if not st.session_state.processing:
            st.session_state.processing = True
            with st.spinner("Processing website content..."):
                st.session_state.content = fetch_website_content(url)
                if st.session_state.content:
                    st.session_state.chunks = split_text(st.session_state.content)
                    st.session_state.chunk_embeddings = create_embeddings(st.session_state.chunks, embedding_model)
            st.session_state.processing = False
else:
    uploaded_file = st.file_uploader("Upload PDF file", type=['pdf'])
    if uploaded_file:
        if not st.session_state.processing:
            st.session_state.processing = True
            with st.spinner("Processing PDF content..."):
                st.session_state.content = extract_pdf_content(uploaded_file)
                if st.session_state.content:
                    st.session_state.chunks = split_text(st.session_state.content)
                    st.session_state.chunk_embeddings = create_embeddings(st.session_state.chunks, embedding_model)
            st.session_state.processing = False

question = st.text_input("Ask a question about the document:", help="Enter your question about the content")

if question and st.session_state.content:
    try:
        with st.spinner("Finding answer..."):
            relevant_chunk = find_most_relevant_chunk(
                question, 
                st.session_state.chunks, 
                st.session_state.chunk_embeddings, 
                embedding_model
            )
            answer = answer_question(question, relevant_chunk, qa_pipeline)
            
            st.markdown("### Answer:")
            st.markdown(f"_{answer}_")
            
            # Add to chat history
            if 'chat_history' not in st.session_state:
                st.session_state.chat_history = []
            st.session_state.chat_history.append((question, answer))
            
    except Exception as e:
        st.error(f"Error processing question: {str(e)}")

# Chat History
if st.session_state.get('chat_history'):
    st.markdown("---")
    st.markdown("### 💬 Chat History")
    for i, (q, a) in enumerate(st.session_state.chat_history, 1):
        with st.expander(f"Q{i}: {q}", expanded=False):
            st.markdown(f"**Answer:** {a}")

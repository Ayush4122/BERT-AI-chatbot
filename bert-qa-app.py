import streamlit as st
import PyPDF2
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForQuestionAnswering
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import io
import re

# Initialize BERT models
@st.cache_resource
def load_models():
    # For encoding text chunks
    encoder = SentenceTransformer('bert-base-nli-mean-tokens')
    # For question answering
    tokenizer = AutoTokenizer.from_pretrained('deepset/bert-base-cased-squad2')
    model = AutoModelForQuestionAnswering.from_pretrained('deepset/bert-base-cased-squad2')
    return encoder, tokenizer, model

# Process PDF and create FAISS index
def process_pdf(pdf_file):
    # Read PDF
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    
    # Split into chunks (simple approach using paragraphs)
    chunks = [chunk.strip() for chunk in text.split('\n\n') if chunk.strip()]
    
    # Create smaller chunks of ~200 words each
    processed_chunks = []
    for chunk in chunks:
        words = chunk.split()
        for i in range(0, len(words), 200):
            processed_chunks.append(' '.join(words[i:i+200]))
    
    return processed_chunks

def create_faiss_index(chunks, encoder):
    # Encode text chunks
    embeddings = encoder.encode(chunks)
    embeddings = np.array(embeddings).astype('float32')
    
    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    return index, embeddings

def get_answer(question, context, tokenizer, model):
    # Prepare input
    inputs = tokenizer(question, context, return_tensors="pt", max_length=512, truncation=True)
    
    # Get answer
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get answer span
    answer_start = torch.argmax(outputs.start_logits)
    answer_end = torch.argmax(outputs.end_logits)
    
    answer = tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(
            inputs["input_ids"][0][answer_start:answer_end+1]
        )
    )
    
    return answer

# Streamlit UI
st.title("PDF Question Answering Chatbot")

# File upload
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file is not None:
    # Load models
    encoder, tokenizer, qa_model = load_models()
    
    # Process PDF if not already processed
    if 'chunks' not in st.session_state:
        with st.spinner('Processing PDF...'):
            chunks = process_pdf(uploaded_file)
            index, embeddings = create_faiss_index(chunks, encoder)
            st.session_state['chunks'] = chunks
            st.session_state['index'] = index
            st.session_state['embeddings'] = embeddings
        st.success('PDF processed successfully!')
    
    # Question input
    question = st.text_input("Ask a question about the PDF:")
    
    if question:
        with st.spinner('Finding answer...'):
            # Encode question
            q_embedding = encoder.encode([question])
            
            # Search similar chunks
            k = 3  # Number of chunks to retrieve
            D, I = st.session_state['index'].search(q_embedding.astype('float32'), k)
            
            # Combine relevant chunks
            context = " ".join([st.session_state['chunks'][i] for i in I[0]])
            
            # Get answer
            answer = get_answer(question, context, tokenizer, qa_model)
            
            # Display answer
            st.write("Answer:", answer)
            
            # Display relevant context (expandable)
            with st.expander("View relevant context"):
                for i in I[0]:
                    st.write(st.session_state['chunks'][i])

# Add requirements info
requirements = """
Requirements:
pip install streamlit PyPDF2 torch transformers sentence-transformers faiss-cpu
"""

with st.expander("View Requirements"):
    st.code(requirements)

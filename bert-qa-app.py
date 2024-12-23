import streamlit as st
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch

# Initialize BERT models
@st.cache_resource
def load_models():
    # Use a smaller BERT model for embeddings
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    # Use a smaller BERT model for question answering
    qa_model = pipeline('question-answering', model='deepset/minilm-uncased-squad2')
    return embedding_model, qa_model

def fetch_website_content(url, max_pages=5):
    def get_links(url):
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        return [a['href'] for a in soup.find_all('a', href=True) if a['href'].startswith(url)]
    
    visited = set()
    to_visit = [url]
    content = ""
    
    while to_visit and len(visited) < max_pages:
        current_url = to_visit.pop(0)
        if current_url not in visited:
            try:
                response = requests.get(current_url)
                soup = BeautifulSoup(response.content, 'html.parser')
                content += soup.get_text() + "\n\n"
                visited.add(current_url)
                to_visit.extend([link for link in get_links(current_url) if link not in visited])
            except Exception as e:
                st.error(f"Error fetching {current_url}: {str(e)}")
    return content

def split_text(text, chunk_size=500):
    # Simple text splitting by sentences
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
    return embedding_model.encode(chunks)

def find_most_relevant_chunk(question, chunks, chunk_embeddings, embedding_model):
    question_embedding = embedding_model.encode([question])[0]
    similarities = cosine_similarity([question_embedding], chunk_embeddings)[0]
    most_similar_idx = np.argmax(similarities)
    return chunks[most_similar_idx]

def answer_question(question, context, qa_pipeline):
    result = qa_pipeline(question=question, context=context)
    return result['answer']

# Streamlit UI
st.title("Website Q&A Chatbot (BERT)")

# Load models
embedding_model, qa_pipeline = load_models()

url = st.text_input("Enter website URL:")
question = st.text_input("Ask a question about the website:")

if url and question:
    with st.spinner("Fetching website content..."):
        content = fetch_website_content(url)
    
    with st.spinner("Processing content..."):
        chunks = split_text(content)
        chunk_embeddings = create_embeddings(chunks, embedding_model)
    
    with st.spinner("Finding relevant content and answering..."):
        relevant_chunk = find_most_relevant_chunk(question, chunks, chunk_embeddings, embedding_model)
        answer = answer_question(question, relevant_chunk, qa_pipeline)
    
    st.write("Answer:", answer)

# Chat History
st.markdown("---")
st.write("Chat History:")
for i, (q, a) in enumerate(st.session_state.get('chat_history', []), 1):
    st.write(f"Q{i}: {q}")
    st.write(f"A{i}: {a}")

if url and question:
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    st.session_state.chat_history.append((question, answer))

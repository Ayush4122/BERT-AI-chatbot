import streamlit as st
import PyPDF2
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForQuestionAnswering
import numpy as np
from numpy import triu
import re
import faiss
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from numpy.linalg import norm

class DocumentProcessor:
    def __init__(self, chunk_size: int = 200, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s.,!?;:]', ' ', text)
        return text.strip()

    def tokenize_text(self, text: str) -> list:
        """Tokenize text into words."""
        return simple_preprocess(text, deacc=True)

    def simple_split_into_sentences(self, text: str) -> list:
        """Split text into sentences using simple rules."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def create_chunks(self, text: str) -> list:
        """Create overlapping chunks of text."""
        sentences = self.simple_split_into_sentences(text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence.split())
            
            if current_length + sentence_length <= self.chunk_size:
                current_chunk.append(sentence)
                current_length += sentence_length
            else:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_length = 0
                current_chunk.append(sentence)
                current_length = sentence_length
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks

    def process_pdf(self, pdf_file) -> list:
        """Process PDF and return cleaned, chunked text."""
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"

            cleaned_text = self.clean_text(text)
            chunks = self.create_chunks(cleaned_text)
            
            return chunks
        except Exception as e:
            st.error(f"Error processing PDF: {str(e)}")
            return None

class Word2VecVectorStore:
    def __init__(self, vector_size=100, window=5, min_count=1):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.model = None
        self.index = None
        self.chunks = None

    def train_word2vec(self, chunks: list):
        """Train Word2Vec model on the chunks."""
        # Tokenize all chunks
        tokenized_chunks = [simple_preprocess(chunk) for chunk in chunks]
        
        # Train Word2Vec model
        self.model = Word2Vec(sentences=tokenized_chunks,
                            vector_size=self.vector_size,
                            window=self.window,
                            min_count=self.min_count,
                            workers=4)
        
    def get_chunk_embedding(self, chunk: str) -> np.ndarray:
        """Get embedding for a chunk of text."""
        tokens = simple_preprocess(chunk)
        token_embeddings = []
        
        for token in tokens:
            if token in self.model.wv:
                token_embeddings.append(self.model.wv[token])
        
        if not token_embeddings:
            return np.zeros(self.vector_size)
        
        # Average word embeddings to get chunk embedding
        chunk_embedding = np.mean(token_embeddings, axis=0)
        # Normalize the embedding
        chunk_embedding = chunk_embedding / norm(chunk_embedding)
        return chunk_embedding

    def create_index(self, chunks: list):
        """Create FAISS index from text chunks."""
        try:
            # Train Word2Vec model first
            self.train_word2vec(chunks)
            
            # Create progress bar
            total_chunks = len(chunks)
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Process chunks and update progress
            embeddings = []
            for i, chunk in enumerate(chunks):
                status_text.text(f'Processing chunk {i+1}/{total_chunks}')
                embedding = self.get_chunk_embedding(chunk)
                embeddings.append(embedding)
                progress_bar.progress((i + 1) / total_chunks)
            
            # Convert to numpy array
            embeddings = np.array(embeddings).astype('float32')
            
            # Create FAISS index
            self.index = faiss.IndexFlatL2(self.vector_size)
            self.index.add(embeddings)
            self.chunks = chunks
            
            # Clean up progress indicators
            status_text.empty()
            progress_bar.empty()
            
        except Exception as e:
            st.error(f"Error creating index: {str(e)}")
            return None

    def search(self, query: str, k: int = 3) -> list:
        """Search for relevant chunks."""
        try:
            query_embedding = self.get_chunk_embedding(query)
            D, I = self.index.search(query_embedding.reshape(1, -1).astype('float32'), k)
            return [(self.chunks[i], score) for i, score in zip(I[0], D[0])]
        except Exception as e:
            st.error(f"Error during search: {str(e)}")
            return []

def main():
    st.title("📚 PDF Question Answering System")
    
    # Initialize models
    if 'processor' not in st.session_state:
        st.session_state.processor = DocumentProcessor()
        st.session_state.vector_store = Word2VecVectorStore()
        st.session_state.qa_model = QuestionAnswerer()
    
    # File upload
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
    
    if uploaded_file is not None:
        # Process PDF if not already processed
        if 'processed_file' not in st.session_state or st.session_state.processed_file != uploaded_file.name:
            with st.spinner('Processing PDF... Please wait.'):
                chunks = st.session_state.processor.process_pdf(uploaded_file)
                if chunks:
                    st.session_state.vector_store.create_index(chunks)
                    st.session_state.processed_file = uploaded_file.name
                    st.success('PDF processed successfully! You can now ask questions.')
        
        # Question input
        question = st.text_input("Ask a question about the PDF:", 
                               help="Type your question here and press Enter")
        
        if question:
            with st.spinner('Finding answer...'):
                results = st.session_state.vector_store.search(question)
                
                if results:
                    context = " ".join([chunk for chunk, _ in results])
                    answer, confidence = st.session_state.qa_model.get_answer(question, context)
                    
                    if answer:
                        st.markdown("### Answer:")
                        st.write(answer)
                        st.progress(confidence)
                        st.caption(f"Confidence: {confidence:.2%}")
                        
                        with st.expander("View source contexts"):
                            for i, (chunk, score) in enumerate(results, 1):
                                st.markdown(f"**Relevant text {i}:**")
                                st.write(chunk)
                                st.caption(f"Relevance score: {1/(1+score):.2%}")
                    else:
                        st.warning("Could not generate an answer. Please try rephrasing your question.")
                else:
                    st.warning("Could not find relevant information in the document.")

if __name__ == "__main__":
    main()

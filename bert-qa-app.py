import streamlit as st
import PyPDF2
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForQuestionAnswering
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Tuple, Dict
import re
from nltk.tokenize import sent_tokenize
import nltk
import faiss
from tqdm import tqdm

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class DocumentProcessor:
    def __init__(self, chunk_size: int = 300, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s.,!?;:]', ' ', text)
        return text.strip()

    def create_chunks(self, text: str) -> List[str]:
        """Create overlapping chunks of text that preserve sentence boundaries."""
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence_length = len(sentence.split())
            
            if current_length + sentence_length <= self.chunk_size:
                current_chunk.append(sentence)
                current_length += sentence_length
            else:
                # Add the current chunk if it's not empty
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                
                # Start new chunk with overlap
                overlap_chunk = []
                overlap_length = 0
                
                # Add sentences from the end of the previous chunk for overlap
                for prev_sentence in current_chunk[::-1]:
                    prev_length = len(prev_sentence.split())
                    if overlap_length + prev_length <= self.chunk_overlap:
                        overlap_chunk.insert(0, prev_sentence)
                        overlap_length += prev_length
                    else:
                        break
                
                current_chunk = overlap_chunk + [sentence]
                current_length = sum(len(s.split()) for s in current_chunk)

        # Add the last chunk if it's not empty
        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks

    def process_pdf(self, pdf_file) -> List[str]:
        """Process PDF and return cleaned, chunked text."""
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()

            # Clean the text
            cleaned_text = self.clean_text(text)
            
            # Create chunks
            chunks = self.create_chunks(cleaned_text)
            
            return chunks
        except Exception as e:
            st.error(f"Error processing PDF: {str(e)}")
            return None

class VectorStore:
    def __init__(self, encoder_model: str = 'sentence-transformers/all-mpnet-base-v2'):
        self.encoder = SentenceTransformer(encoder_model)
        self.index = None
        self.chunks = None
        self.embeddings = None

    def create_index(self, chunks: List[str]):
        """Create FAISS index from text chunks."""
        # Create embeddings with progress bar
        embeddings = []
        for chunk in tqdm(chunks, desc="Creating embeddings"):
            embedding = self.encoder.encode(chunk)
            embeddings.append(embedding)
        
        embeddings = np.array(embeddings).astype('float32')
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        
        self.index = index
        self.chunks = chunks
        self.embeddings = embeddings

    def search(self, query: str, k: int = 3) -> List[Tuple[str, float]]:
        """Search for relevant chunks."""
        query_embedding = self.encoder.encode([query])
        D, I = self.index.search(query_embedding.astype('float32'), k)
        
        results = []
        for score, idx in zip(D[0], I[0]):
            results.append((self.chunks[idx], float(score)))
        
        return results

class QuestionAnswerer:
    def __init__(self, model_name: str = 'deepset/bert-base-cased-squad2'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        self.max_length = 512

    def get_answer(self, question: str, context: str) -> Tuple[str, float]:
        """Get answer from context with confidence score."""
        inputs = self.tokenizer(
            question,
            context,
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        )

        with torch.no_grad():
            outputs = self.model(**inputs)

        start_scores = outputs.start_logits
        end_scores = outputs.end_logits
        
        start_idx = torch.argmax(start_scores)
        end_idx = torch.argmax(end_scores)
        
        # Calculate confidence score
        start_score = torch.softmax(start_scores, dim=1).max().item()
        end_score = torch.softmax(end_scores, dim=1).max().item()
        confidence = (start_score + end_score) / 2

        # Get answer tokens
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        answer = self.tokenizer.convert_tokens_to_string(tokens[start_idx:end_idx+1])

        return answer, confidence

def initialize_session_state():
    """Initialize session state variables."""
    if 'processor' not in st.session_state:
        st.session_state.processor = DocumentProcessor()
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = VectorStore()
    if 'qa_model' not in st.session_state:
        st.session_state.qa_model = QuestionAnswerer()
    if 'processed_files' not in st.session_state:
        st.session_state.processed_files = set()

def main():
    st.title("📚 Enhanced PDF Question Answering System")
    
    # Initialize session state
    initialize_session_state()
    
    # File upload
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
    
    if uploaded_file is not None:
        # Check if file was already processed
        file_name = uploaded_file.name
        
        if file_name not in st.session_state.processed_files:
            with st.spinner('Processing PDF... This may take a minute.'):
                # Process PDF
                chunks = st.session_state.processor.process_pdf(uploaded_file)
                if chunks:
                    # Create embeddings and index
                    st.session_state.vector_store.create_index(chunks)
                    st.session_state.processed_files.add(file_name)
                    st.success('PDF processed successfully! You can now ask questions.')
        
        # Question input
        question = st.text_input("Ask a question about the PDF:", 
                               help="Type your question here and press Enter")
        
        if question:
            with st.spinner('Finding answer...'):
                # Get relevant chunks
                results = st.session_state.vector_store.search(question)
                
                # Combine contexts and get answer
                context = " ".join([chunk for chunk, score in results])
                answer, confidence = st.session_state.qa_model.get_answer(question, context)
                
                # Display answer with confidence
                st.markdown("### Answer:")
                st.write(answer)
                st.progress(confidence)
                st.caption(f"Confidence: {confidence:.2%}")
                
                # Show relevant contexts
                with st.expander("View source contexts"):
                    for i, (chunk, score) in enumerate(results, 1):
                        st.markdown(f"**Relevant text {i}:**")
                        st.write(chunk)
                        st.caption(f"Relevance score: {1/(1+score):.2%}")

if __name__ == "__main__":
    main()

import streamlit as st
import PyPDF2
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForQuestionAnswering
from sentence_transformers import SentenceTransformer
import numpy as np
import re
import faiss

class DocumentProcessor:
    def __init__(self, chunk_size: int = 200, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s.,!?;:]', ' ', text)
        return text.strip()

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

class VectorStore:
    def __init__(self):
        self.encoder = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        self.index = None
        self.chunks = None

    def create_index(self, chunks: list):
        """Create FAISS index from text chunks."""
        try:
            # Create progress bar
            total_chunks = len(chunks)
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Process chunks and update progress
            embeddings = []
            for i, chunk in enumerate(chunks):
                status_text.text(f'Processing chunk {i+1}/{total_chunks}')
                embedding = self.encoder.encode(chunk)
                embeddings.append(embedding)
                progress_bar.progress((i + 1) / total_chunks)
            
            # Convert to numpy array
            embeddings = np.array(embeddings).astype('float32')
            
            # Create FAISS index
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
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
            query_vector = self.encoder.encode([query])
            D, I = self.index.search(query_vector.astype('float32'), k)
            return [(self.chunks[i], score) for i, score in zip(I[0], D[0])]
        except Exception as e:
            st.error(f"Error during search: {str(e)}")
            return []

class QuestionAnswerer:
    def __init__(self):
        model_name = 'deepset/bert-base-cased-squad2'
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)

    def get_answer(self, question: str, context: str) -> tuple:
        """Get answer from context with confidence score."""
        try:
            inputs = self.tokenizer(
                question,
                context,
                max_length=512,
                truncation=True,
                return_tensors="pt"
            )

            with torch.no_grad():
                outputs = self.model(**inputs)

            start_scores = outputs.start_logits
            end_scores = outputs.end_logits
            
            start_idx = torch.argmax(start_scores)
            end_idx = torch.argmax(end_scores)
            
            start_score = torch.softmax(start_scores, dim=1).max().item()
            end_score = torch.softmax(end_scores, dim=1).max().item()
            confidence = (start_score + end_score) / 2

            tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
            answer = self.tokenizer.convert_tokens_to_string(tokens[start_idx:end_idx+1])

            return answer, confidence
        except Exception as e:
            st.error(f"Error generating answer: {str(e)}")
            return "", 0.0

def main():
    st.title("📚 PDF Question Answering System")
    
    # Initialize models
    if 'processor' not in st.session_state:
        st.session_state.processor = DocumentProcessor()
        st.session_state.vector_store = VectorStore()
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

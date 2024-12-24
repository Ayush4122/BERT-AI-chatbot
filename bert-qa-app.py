import streamlit as st
import PyPDF2
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import numpy as np
from numpy.linalg import norm
import re
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document

# Document Processor Class
class DocumentProcessor:
    def __init__(self, chunk_size=200, chunk_overlap=50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def clean_text(self, text):
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s.,!?;:]', ' ', text)
        return text.strip()

    def split_into_sentences(self, text):
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def create_chunks(self, text):
        sentences = self.split_into_sentences(text)
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
                current_chunk = [sentence]
                current_length = sentence_length

        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks

    def process_pdf(self, pdf_file):
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = "".join(page.extract_text() for page in pdf_reader.pages)
            cleaned_text = self.clean_text(text)
            return self.create_chunks(cleaned_text)
        except Exception as e:
            st.error(f"Error processing PDF: {e}")
            return None

# LangChain-based Vector Store
class LangChainVectorStore:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings()
        self.vector_store = None

    def create_index(self, chunks):
        try:
            documents = [Document(page_content=chunk) for chunk in chunks]
            self.vector_store = FAISS.from_documents(documents, self.embeddings)
        except Exception as e:
            st.error(f"Error creating index: {e}")

    def search(self, query, k=3):
        try:
            return self.vector_store.similarity_search_with_score(query, k)
        except Exception as e:
            st.error(f"Error during search: {e}")
            return []

# Question Answering Model
class QAEngine:
    def __init__(self, model_name="distilbert-base-uncased-distilled-squad"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)

    def get_answer(self, question, context):
        inputs = self.tokenizer(question, context, return_tensors="pt", truncation=True, max_length=512)
        outputs = self.model(**inputs)
        start_idx = torch.argmax(outputs.start_logits)
        end_idx = torch.argmax(outputs.end_logits) + 1
        answer = self.tokenizer.decode(inputs.input_ids[0][start_idx:end_idx], skip_special_tokens=True)
        confidence = torch.nn.functional.softmax(outputs.start_logits + outputs.end_logits, dim=1).max().item()
        return answer, confidence

# Streamlit App
def main():
    st.title("📚 PDF Question Answering System")

    # Initialize models
    if 'processor' not in st.session_state:
        st.session_state.processor = DocumentProcessor()
        st.session_state.vector_store = LangChainVectorStore()
        st.session_state.qa_engine = QAEngine()

    # File upload
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

    if uploaded_file is not None:
        if 'processed_file' not in st.session_state or st.session_state.processed_file != uploaded_file.name:
            with st.spinner('Processing PDF...'):
                chunks = st.session_state.processor.process_pdf(uploaded_file)
                if chunks:
                    st.session_state.vector_store.create_index(chunks)
                    st.session_state.processed_file = uploaded_file.name
                    st.success("PDF processed successfully! Ask your questions below.")

        # Question input
        question = st.text_input("Ask a question about the PDF:")

        if question:
            with st.spinner("Finding answer..."):
                results = st.session_state.vector_store.search(question)
                if results:
                    context = " ".join([doc.page_content for doc, _ in results])
                    answer, confidence = st.session_state.qa_engine.get_answer(question, context)

                    if answer:
                        st.markdown("### Answer:")
                        st.write(answer)
                        st.progress(confidence)

                        with st.expander("View source contexts"):
                            for i, (doc, score) in enumerate(results, 1):
                                st.markdown(f"**Relevant text {i}:**")
                                st.write(doc.page_content)
                                st.caption(f"Relevance score: {score:.2f}")
                    else:
                        st.warning("Could not generate an answer. Try rephrasing your question.")
                else:
                    st.warning("No relevant information found in the document.")

if __name__ == "__main__":
    main()

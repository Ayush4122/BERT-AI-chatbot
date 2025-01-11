import streamlit as st
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders import JSONLoader
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import HuggingFaceHub
import os

# Load BERT model and tokenizer
def load_bert_model():
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model = SentenceTransformer(model_name)
    return model

# Load documents from a PDF or JSON file
def load_documents(file):
    if file.type == "application/pdf":
        loader = PyMuPDFLoader(file.name)
    elif file.type == "application/json":
        # Define jq_schema or pass it as needed
        jq_schema = "$.data"  # For example, specify a path to the relevant data in the JSON
        loader = JSONLoader(file.name, jq_schema=jq_schema)
    else:
        st.error("Unsupported file type. Please upload a PDF or JSON file.")
        return None
    return loader.load()
# Split documents into chunks
def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return text_splitter.split_documents(documents)

# Build FAISS vector store
def build_vector_store(documents, model):
    embeddings = model.encode([doc.page_content for doc in documents])
    faiss_store = FAISS.from_documents(documents, embeddings)
    return faiss_store

# Set up the Retrieval-based QA chain
def setup_qa_chain(vector_store):
    llm = HuggingFaceHub(repo_id="google/flan-t5-base")  # Replace with a suitable model if needed
    qa_chain = RetrievalQA(llm=llm, retriever=vector_store.as_retriever())
    return qa_chain

# Streamlit UI
def main():
    st.title("RAG Implementation with BERT, FAISS, and Streamlit")

    uploaded_file = st.file_uploader("Upload a PDF or JSON file", type=["pdf", "json"])

    if uploaded_file is not None:
        with st.spinner("Loading and processing the file..."):
            documents = load_documents(uploaded_file)
            if documents:
                model = load_bert_model()
                split_docs = split_documents(documents)
                vector_store = build_vector_store(split_docs, model)
                qa_chain = setup_qa_chain(vector_store)
                st.success("File processed and system ready!")

                query = st.text_input("Enter your question:")
                if query:
                    with st.spinner("Fetching answer..."):
                        response = qa_chain.run(query)
                        st.write("Answer:", response)

if __name__ == "__main__":
    main()

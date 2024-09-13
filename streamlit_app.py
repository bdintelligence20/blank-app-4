import os
import pandas as pd
import streamlit as st
import sqlite3
import logging
import torch

from langchain_community.vectorstores import FAISS  # Updated import from langchain-community
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain

from langchain.cache import SQLiteCache
import langchain

# Set up caching for embeddings to prevent redundant computations
langchain.llm_cache = SQLiteCache(database_path=".langchain_cache.db")

# Custom Embeddings Class for NV-Embed-v2
from langchain.embeddings.base import Embeddings
from transformers import AutoTokenizer, AutoModel
from typing import List

class NVEmbedV2Embeddings(Embeddings):
    def __init__(self, model_name: str = "nvidia/NV-Embed-v2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        all_embeddings = []
        batch_size = 16  # Adjust based on your system's capacity
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            inputs = self.tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)
            all_embeddings.extend(embeddings.cpu().numpy())
        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

# Initialize embeddings using the custom class
embeddings = NVEmbedV2Embeddings()

# Initialize the language model (e.g., OpenAI's GPT-4)
openai_api_key = st.secrets["general"]["OPENAI_API_KEY"]

llm = ChatOpenAI(
    openai_api_key=openai_api_key,
    model_name="gpt-4",
    temperature=0.7,
    max_tokens=2000
)

# Database setup for persistent memory
conn = sqlite3.connect('chat_history.db')
c = conn.cursor()

# Create table for chat history
c.execute('''CREATE TABLE IF NOT EXISTS chat_history
             (user_input TEXT, assistant_response TEXT)''')
conn.commit()

# Function to load documents (optimized for large files)
def load_document(file):
    file_extension = os.path.splitext(file.name)[1].lower()
    if file_extension == ".pdf":
        from PyPDF2 import PdfReader
        reader = PdfReader(file)
        return "\n".join(page.extract_text() for page in reader.pages)
    elif file_extension == ".csv":
        df = pd.read_csv(file)
        return df.to_string(index=False)
    elif file_extension in [".xls", ".xlsx"]:
        df = pd.read_excel(file)
        return df.to_string(index=False)
    else:
        return file.read().decode('utf-8')

# Function to create the vector store with the new embeddings
def create_vector_store(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = []
    for document in documents:
        splits = text_splitter.split_text(document)
        texts.extend(splits)
    with st.spinner('Computing embeddings...'):
        embeddings_list = embeddings.embed_documents(texts)
    # Using FAISS from langchain-community
    vector_store = FAISS.from_embeddings(embeddings=embeddings_list, texts=texts)
    return vector_store

# Function to save chat history to the database
def save_chat_history(user_input, assistant_response):
    c.execute("INSERT INTO chat_history (user_input, assistant_response) VALUES (?, ?)",
              (user_input, assistant_response))
    conn.commit()

# Function to load chat history from the database
def load_chat_history():
    c.execute("SELECT user_input, assistant_response FROM chat_history")
    return c.fetchall()

# Streamlit app setup
st.title("Optimized RAG Chatbot with NVIDIA's NV-Embed-v2 and FAISS")

# Upload documents
uploaded_files = st.file_uploader(
    "Upload your documents (txt, pdf, csv, xlsx)",
    accept_multiple_files=True,
    type=["txt", "pdf", "csv", "xlsx"]
)

# Advanced prompt template
prompt_template = PromptTemplate(
    input_variables=["history", "question"],
    template="""
    You are an expert assistant with access to the following conversation history:
    {history}

    Answer the following question:
    {question}

    Provide a comprehensive and informative response.
    """
)

# Chat functionality
if uploaded_files:
    # Load and process documents
    with st.spinner('Loading and processing documents...'):
        documents = [load_document(file) for file in uploaded_files]

    # Create or load vector store
    VECTOR_STORE_PATH = "vector_store"
    if os.path.exists(VECTOR_STORE_PATH):
        # Load existing vector store
        vector_store = FAISS.load_local(VECTOR_STORE_PATH, embeddings)
    else:
        # Create a new vector store
        vector_store = create_vector_store(documents)
        vector_store.save_local(VECTOR_STORE_PATH)

    # Create conversational chain
    retriever = vector_store.as_retriever(search_type="similarity", k=5)
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        condense_question_prompt=prompt_template,
    )

    # Initialize chat history from the database
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = load_chat_history()

    # Chat interface
    user_input = st.chat_input("Ask a question about your documents:")
    if user_input:
        try:
            response = qa_chain({
                "question": user_input,
                "chat_history": st.session_state["chat_history"]
            })
            assistant_response = response['answer']
            st.session_state["chat_history"].append((user_input, assistant_response))

            # Save to the database
            save_chat_history(user_input, assistant_response)
        except Exception as e:
            assistant_response = "I'm sorry, I encountered an error processing your request."
            st.error("An error occurred while processing your request.")
            logging.error(e)

    # Display chat history
    for question, answer in st.session_state["chat_history"]:
        st.chat_message("user").write(question)
        st.chat_message("assistant").write(answer)

    # Additional functionality - Summarization model
    st.header("Summarization")
    summarize_button = st.button("Summarize Documents")

    if summarize_button:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.create_documents(documents)
        summary_chain = load_summarize_chain(llm, chain_type="map_reduce")
        with st.spinner('Generating summary...'):
            summary = summary_chain.run(docs)
        st.write("Summary of uploaded documents:")
        st.write(summary)

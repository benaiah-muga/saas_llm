from dotenv import load_dotenv
load_dotenv()

import os
import requests
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Pinecone as LC_Pinecone  
from langchain.chains import RetrievalQA
from jina_embeddings import JinaEmbeddings

os.environ["USER_AGENT"] = "saas_llm/0.1"

# --- Pinecone Index Initialization ---
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
index_name = os.getenv("PINECONE_INDEX")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY is not set in the environment!")
    
pc = Pinecone(api_key=PINECONE_API_KEY)
existing_indexes = pc.list_indexes().names()
if index_name not in existing_indexes:
    spec = ServerlessSpec(cloud=os.getenv("PINECONE_CLOUD", "aws"), region=PINECONE_ENV)
    pc.create_index(index_name, dimension=1024, metric="cosine", spec=spec)
index = pc.Index(index_name)

# --- Step 1: Ingest Webpage Content ---
def ingest_webpage(url: str):
    loader = WebBaseLoader(url)
    docs = loader.load()  # Returns a list of Document objects
    return docs

# --- Step 2: Split Documents into Chunks ---
def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    return chunks

# --- Step 4: Create Embeddings and Vector Store Using JinaEmbeddings ---
def create_vector_store(docs):
    chunks = split_documents(docs)
    embeddings = JinaEmbeddings()  # Uses API key from os.getenv("JINA_API_KEY")
    vector_store = LC_Pinecone.from_documents(chunks, embeddings, index_name=index_name)
    return vector_store

# --- Step 5: Initialize the Chat Model using Mistral ---
def get_chat_model():
    from langchain_mistralai.chat_models import ChatMistralAI
    return ChatMistralAI(
        api_key=os.getenv("MISTRAL_API_KEY"),
        model="mistral-large-latest",
        temperature=0,
        max_retries=2
    )

model = get_chat_model()

def build_qa_chain(vector_store):
    return RetrievalQA.from_llm(
        llm=model,
        retriever=vector_store.as_retriever(),
        chain_type="stuff"  # "stuff" concatenates context chunks into a prompt
    )

# --- Step 6: Generate RAG-Based Answers ---
def generate_rag_answer(qa_chain, query: str):
    result = qa_chain.call({"query": query})
    return result["result"]

# --- Example Usage for Command-Line Testing ---
if __name__ == "__main__":
    url = "https://www.postman.com/explore"  # Replace with the customer-provided URL
    docs = ingest_webpage(url)
    vector_store = create_vector_store(docs)
    qa_chain = build_qa_chain(vector_store)
    query = "What are the main features described on the webpage?"
    answer = generate_rag_answer(qa_chain, query)
    print("RAG Answer:", answer)


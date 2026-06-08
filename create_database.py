# ============================================================
# create_database.py
# Loads a TD financial report PDF, splits it into chunks,
# and stores them in a ChromaDB vector database using
# Azure OpenAI embeddings
# ============================================================

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import AzureOpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
import os
import shutil

# Load environment variables from .env file
load_dotenv()

# TD financial report PDF
DATA_PATH = r"data/TD-2026-q2-earnings-report.pdf"

# Path to the ChromaDB vector store
CHROMA_PATH = r"chroma_db"


def main():
    generate_data_store()

# full pipeline: load -> split -> store.
def generate_data_store():
    
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)


def load_documents():
    loader = PyPDFLoader(DATA_PATH)
    documents = loader.load()
    return documents

# Splits PDF pages into smaller chunks for embedding.
def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    # Preview a sample chunk for debugging
    print(chunks[10].page_content)
    print(chunks[10].metadata)
    return chunks

#  Embeds document chunks using Azure OpenAI and saves them to ChromaDB vector store.
def save_to_chroma(chunks: list[Document]):
    # Clear existing ChromaDB to avoid duplicate entries
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # Initialize Azure OpenAI embeddings using environment variables
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION")
    )

    # Create ChromaDB vector store from document chunks and embeddings
    db = Chroma.from_documents(
        chunks, embeddings, persist_directory=CHROMA_PATH
    )
   
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")


if __name__ == "__main__":
    main()
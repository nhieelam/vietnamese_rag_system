import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
# File loader import removed - not needed in ingest_service
# from app.services.file_loader import extract_text_from_file
from app.config import AppConfig

DB_DIRECTORY = AppConfig.DB_DIRECTORY

def ingest_file(raw_text):
    
    if not raw_text.strip():
        print("No text found !")
        return

    print("Text extracted. Splitting into chunks...")

    text_splitter = RecursiveCharacterTextSplitter(
        AppConfig.CHUNK_SIZE,
        AppConfig.CHUNK_OVERLAP
    )
    docs = text_splitter.create_documents([raw_text])

    print("Saving to ChromaDB...")
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    
    # We use 'add_documents' to append to the existing DB, not overwrite it
    vector_db = Chroma(
        persist_directory=DB_DIRECTORY, 
        embedding_function=embedding_model
    )
    vector_db.add_documents(docs)
    
    print(f"Success! Added {len(docs)} chunks to the database.")
    


import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

# 1. SETUP: Define your text data
# (In a real app, this text comes from your OCR results)
raw_text = """
Chính sách an toàn lao động năm 2024:
1. Tất cả nhân viên phải đội mũ bảo hộ khi vào công trường.
2. Thời gian làm việc không quá 8 tiếng mỗi ngày.
3. Báo cáo tai nạn phải được gửi trong vòng 24 giờ.
4. Mức phạt vi phạm quy định an toàn là 500.000 VNĐ.
"""

def create_vector_db():
    print("🔄 Starting to build Vector Database...")


    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    
    docs = text_splitter.create_documents([raw_text])
    print(f"✅ Split text into {len(docs)} chunks.")

    # 3. EMBEDDING: Load a model that understands Vietnamese
    # We use 'paraphrase-multilingual-MiniLM-L12-v2' which supports Vietnamese
    print("🔄 Loading Embedding Model (this may take a minute)...")
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    persist_directory = "./chroma_db"
    
    vector_db = Chroma.from_documents(
        documents=docs,
        embedding=embedding_model,
        persist_directory=persist_directory
    )
    
    print("Vector Database created successfully in './chroma_db'")
    return vector_db

if __name__ == "__main__":
    create_vector_db()
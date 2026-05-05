import os
import shutil
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# --- Configuration ---
CHROMA_PATH = os.path.join("data", "processed", "chroma_db")
RAW_DATA_PATH = os.path.join("data", "raw")

def initialize_embedding_model():
    """Loads the open-source embedding model."""
    print("Loading embedding model (all-MiniLM-L6-v2)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return embeddings

def build_vector_store(chunks):
    """
    Takes the text chunks, converts them to vectors using the embedding model,
    and saves them permanently into ChromaDB.
    """
    # 1. Clear out any old database to prevent duplicates during testing
    if os.path.exists(CHROMA_PATH):
        print("Clearing old vector database...")
        shutil.rmtree(CHROMA_PATH)
    
    # 2. Get the embedding model
    embeddings = initialize_embedding_model()
    
    print(f"Embedding {len(chunks)} chunks and building ChromaDB...")
    print("This might take a few minutes depending on your CPU...")
    
    # 3. Create the database and save it to disk
    db = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        persist_directory=CHROMA_PATH
    )
    
    print(f"Vector database successfully saved to {CHROMA_PATH}")
    return db

# Test the pipeline
if __name__ == "__main__":
    import sys
    # Ensure Python can find our previous scripts
    sys.path.append(os.path.abspath("."))
    
    from src.data.ingestion import ingest_pdfs
    from src.data.transformation import chunk_documents
    
    print("--- 1. Ingesting ---")
    docs = ingest_pdfs(RAW_DATA_PATH)
    
    print("\n--- 2. Chunking ---")
    chunks = chunk_documents(docs)
    
    print("\n--- 3. Embedding and Storing ---")
    build_vector_store(chunks)
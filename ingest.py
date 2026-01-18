"""
Document Ingestion Pipeline for FitRAG

This module handles loading, chunking, and embedding fitness documents into ChromaDB.
Supports PDF and text file formats with recursive character text splitting.
"""

import os
from pathlib import Path
from typing import List
import sys

from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

# Embedding options - uncomment the one you're using
# Option A: Voyage AI (Recommended - excellent quality, free tier)
from langchain_community.embeddings import VoyageEmbeddings

# Option B: Cohere (Alternative - free tier available)
# from langchain_community.embeddings import CohereEmbeddings

# Option C: HuggingFace (Local - no API needed)
# from langchain_community.embeddings import HuggingFaceEmbeddings


# Constants
DATA_DIR = "./data"
CHROMA_PERSIST_DIR = "./chroma_db"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBEDDING_MODEL = "voyage-2"  # Change if using different provider


def load_documents(data_dir: str) -> List:
    """
    Load all PDF and text documents from the specified directory.
    
    Args:
        data_dir: Path to directory containing documents
        
    Returns:
        List of loaded documents
        
    Raises:
        FileNotFoundError: If data directory doesn't exist
    """
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    documents = []
    
    # Load text files
    try:
        text_loader = DirectoryLoader(
            data_dir,
            glob="**/*.txt",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
            show_progress=True
        )
        text_docs = text_loader.load()
        documents.extend(text_docs)
        print(f"✓ Loaded {len(text_docs)} text file(s)")
    except Exception as e:
        print(f"Warning: Error loading text files: {e}")
    
    # Load PDF files
    try:
        pdf_files = list(Path(data_dir).rglob("*.pdf"))
        for pdf_file in pdf_files:
            try:
                loader = PyPDFLoader(str(pdf_file))
                pdf_docs = loader.load()
                documents.extend(pdf_docs)
                print(f"✓ Loaded {len(pdf_docs)} page(s) from {pdf_file.name}")
            except Exception as e:
                print(f"Warning: Error loading {pdf_file.name}: {e}")
    except Exception as e:
        print(f"Warning: Error scanning for PDFs: {e}")
    
    if not documents:
        raise ValueError(f"No documents found in {data_dir}")
    
    return documents


def chunk_documents(documents: List, chunk_size: int = CHUNK_SIZE, 
                    chunk_overlap: int = CHUNK_OVERLAP) -> List:
    """
    Split documents into chunks using recursive character text splitter.
    
    Uses hierarchical separators to maintain semantic coherence.
    Overlap prevents context loss at chunk boundaries.
    
    Args:
        documents: List of documents to chunk
        chunk_size: Maximum size of each chunk
        chunk_overlap: Number of overlapping characters between chunks
        
    Returns:
        List of chunked documents
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],  # Hierarchical splitting
        is_separator_regex=False
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"✓ Created {len(chunks)} chunks from {len(documents)} document(s)")
    print(f"  - Chunk size: {chunk_size}")
    print(f"  - Chunk overlap: {chunk_overlap}")
    
    return chunks


def create_embeddings():
    """
    Initialize the embedding model based on provider choice.
    
    Returns:
        Configured embedding model instance
        
    Raises:
        ValueError: If required API key is not found
    """
    # Option A: Voyage AI (Recommended)
    api_key = os.getenv("VOYAGE_API_KEY")
    if not api_key:
        raise ValueError(
            "VOYAGE_API_KEY not found in environment variables. "
            "Please add it to your .env file."
        )
    return VoyageEmbeddings(
        voyage_api_key=api_key,
        model=EMBEDDING_MODEL
    )
    
    # Option B: Cohere (uncomment if using)
    # api_key = os.getenv("COHERE_API_KEY")
    # if not api_key:
    #     raise ValueError("COHERE_API_KEY not found in environment variables")
    # return CohereEmbeddings(
    #     cohere_api_key=api_key,
    #     model="embed-english-v3.0"
    # )
    
    # Option C: HuggingFace (uncomment if using - no API key needed)
    # return HuggingFaceEmbeddings(
    #     model_name="sentence-transformers/all-mpnet-base-v2"
    # )


def create_vectorstore(chunks: List, embeddings, persist_dir: str = CHROMA_PERSIST_DIR):
    """
    Create and persist ChromaDB vector store from document chunks.
    
    Args:
        chunks: List of chunked documents
        embeddings: Embedding model instance
        persist_dir: Directory to persist the vector database
        
    Returns:
        Chroma vectorstore instance
    """
    print(f"\n⚙️  Creating vector embeddings and storing in ChromaDB...")
    print(f"   This may take a few minutes depending on document size...")
    
    # Create vector store with persistence
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    
    print(f"✓ Vector store created and persisted to {persist_dir}")
    print(f"  - Total vectors: {len(chunks)}")
    
    return vectorstore


def main():
    """
    Main ingestion pipeline execution.
    
    Workflow:
        1. Load environment variables
        2. Load documents from data directory
        3. Chunk documents with overlap strategy
        4. Initialize embedding model
        5. Create and persist vector store
    """
    print("=" * 60)
    print("🏋️  FitRAG Document Ingestion Pipeline")
    print("=" * 60)
    
    try:
        # Load environment variables
        load_dotenv()
        print("\n📋 Step 1: Loading documents...")
        documents = load_documents(DATA_DIR)
        
        print(f"\n✂️  Step 2: Chunking documents...")
        chunks = chunk_documents(documents)
        
        print(f"\n🔢 Step 3: Initializing embeddings...")
        embeddings = create_embeddings()
        print(f"✓ Using {EMBEDDING_MODEL} embedding model")
        
        print(f"\n💾 Step 4: Creating vector store...")
        vectorstore = create_vectorstore(chunks, embeddings)
        
        print("\n" + "=" * 60)
        print("✅ Ingestion Complete!")
        print("=" * 60)
        print(f"📊 Summary:")
        print(f"   - Documents loaded: {len(documents)}")
        print(f"   - Chunks created: {len(chunks)}")
        print(f"   - Vector store: {CHROMA_PERSIST_DIR}")
        print(f"\n🚀 Ready to run: streamlit run app.py")
        print("=" * 60)
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("💡 Tip: Make sure the './data' directory exists with some documents")
        sys.exit(1)
    except ValueError as e:
        print(f"\n❌ Error: {e}")
        print("💡 Tip: Check your .env file has the required API keys")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

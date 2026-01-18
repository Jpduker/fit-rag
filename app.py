"""
FitRAG - AI Fitness Coach Streamlit Application

Production-ready RAG interface for fitness and nutrition questions.
Uses Claude 3.5 Sonnet with ChromaDB vector retrieval.
"""

import os
import streamlit as st
from dotenv import load_dotenv
from typing import Dict, Any

from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Embedding options - MUST match the one used in ingest.py
from langchain_community.embeddings import VoyageEmbeddings
# from langchain_community.embeddings import CohereEmbeddings
# from langchain_community.embeddings import HuggingFaceEmbeddings


# Constants
CHROMA_PERSIST_DIR = "./chroma_db"
EMBEDDING_MODEL = "voyage-2"
LLM_MODEL = "gpt-4"  # OpenAI GPT-4
LLM_TEMPERATURE = 0
RETRIEVAL_K = 4


# Custom prompt template for fitness coaching
CUSTOM_PROMPT_TEMPLATE = """You are an expert AI fitness coach with deep knowledge of exercise science, nutrition, and evidence-based training methodologies. 

Use the following pieces of context to answer the question at the end. Always provide evidence-based answers and cite your sources from the context.

Guidelines:
- If you don't know the answer based on the context, say so honestly
- Always cite which sources support your answer
- Provide specific, actionable advice when possible
- Include relevant scientific principles or research when mentioned in the context
- Be encouraging and supportive in your tone

Context:
{context}

Question: {question}

Evidence-based Answer:"""


def initialize_embeddings():
    """
    Initialize the embedding model (must match ingest.py configuration).
    
    Returns:
        Configured embedding model instance
    """
    # Option A: Voyage AI
    return VoyageEmbeddings(
        voyage_api_key=os.getenv("VOYAGE_API_KEY"),
        model=EMBEDDING_MODEL
    )
    
    # Option B: Cohere (uncomment if using)
    # return CohereEmbeddings(
    #     cohere_api_key=os.getenv("COHERE_API_KEY"),
    #     model="embed-english-v3.0"
    # )
    
    # Option C: HuggingFace (uncomment if using)
    # return HuggingFaceEmbeddings(
    #     model_name="sentence-transformers/all-mpnet-base-v2"
    # )


@st.cache_resource
def load_rag_chain():
    """
    Load and cache the RAG chain with ChromaDB and Claude.
    
    Uses @st.cache_resource to avoid reloading on every interaction.
    
    Returns:
        Tuple of (RetrievalQA chain, vectorstore, retriever)
        
    Raises:
        FileNotFoundError: If ChromaDB directory doesn't exist
        ValueError: If API keys are missing
    """
    # Validate environment
    if not os.path.exists(CHROMA_PERSIST_DIR):
        raise FileNotFoundError(
            f"ChromaDB not found at {CHROMA_PERSIST_DIR}. "
            "Please run 'python ingest.py' first."
        )
    
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    
    # Initialize embeddings
    embeddings = initialize_embeddings()
    
    # Load vector store
    vectorstore = Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=embeddings
    )
    
    # Configure retriever
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": RETRIEVAL_K}
    )
    
    # Initialize LLM
    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        max_tokens=2048,
        openai_api_key=openai_key
    )
    
    # Create custom prompt
    prompt = PromptTemplate(
        template=CUSTOM_PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )
    
    # Create RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # All retrieved docs stuffed into prompt
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    
    return qa_chain, vectorstore, retriever


def format_source_documents(source_docs: list) -> str:
    """
    Format source documents for display with metadata.
    
    Args:
        source_docs: List of retrieved Document objects
        
    Returns:
        Formatted markdown string with sources
    """
    if not source_docs:
        return "*No sources available*"
    
    formatted = ""
    for i, doc in enumerate(source_docs, 1):
        content = doc.page_content.strip()
        metadata = doc.metadata
        
        # Extract metadata
        source = metadata.get('source', 'Unknown')
        filename = os.path.basename(source)
        page = metadata.get('page', 'N/A')
        
        # Format source block
        formatted += f"**Source {i}:** `{filename}`"
        if page != 'N/A':
            formatted += f" (Page {page})"
        formatted += "\n\n"
        formatted += f"> {content[:300]}{'...' if len(content) > 300 else ''}\n\n"
        formatted += "---\n\n"
    
    return formatted


def main():
    """
    Main Streamlit application logic.
    """
    # Page configuration
    st.set_page_config(
        page_title="FitRAG - AI Fitness Coach",
        page_icon="💪",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Load environment variables
    load_dotenv()
    
    # Header
    st.title("💪 FitRAG - AI Fitness Coach")
    st.markdown(
        "Ask me anything about fitness, nutrition, and training! "
        "I'll provide evidence-based answers with citations from trusted sources."
    )
    
    # Sidebar
    with st.sidebar:
        st.header("📚 How FitRAG Works")
        st.markdown("""
        **RAG Pipeline Steps:**
        
        1. **📄 Document Loading**  
           Load fitness PDFs and text files
        
        2. **✂️ Chunking**  
           Split into 1000-char chunks with 200-char overlap
        
        3. **🔢 Embedding**  
           Convert to vector representations using Voyage AI
        
        4. **💾 Storage**  
           Store in ChromaDB vector database
        
        5. **🔍 Retrieval**  
           Semantic similarity search (top-4 chunks)
        
        6. **🤖 Generation**  
           Claude 3.5 Sonnet generates answer with citations
        """)
        
        st.divider()
        
        # Knowledge base stats
        try:
            _, vectorstore, _ = load_rag_chain()
            collection = vectorstore._collection
            chunk_count = collection.count()
            
            st.header("📊 Knowledge Base Stats")
            st.metric("Total Chunks", chunk_count)
            st.metric("Embedding Model", EMBEDDING_MODEL)
            st.metric("LLM Model", "Claude 3.5 Sonnet")
            st.metric("Retrieval Count", f"Top {RETRIEVAL_K}")
            
        except Exception as e:
            st.error(f"⚠️ Error loading vector store: {str(e)}")
            st.info("💡 Run `python ingest.py` to create the knowledge base")
            return
    
    # Main content area
    st.divider()
    
    # Example questions
    with st.expander("💡 Example Questions", expanded=False):
        st.markdown("""
        - What's the optimal protein intake for muscle building?
        - Explain progressive overload and how to implement it
        - How does sleep affect muscle recovery and growth?
        - What's the difference between cardio and strength training benefits?
        - When should I eat protein for maximum muscle synthesis?
        """)
    
    # User input
    user_question = st.text_input(
        "Ask your fitness question:",
        placeholder="e.g., How much protein should I consume daily for muscle gain?",
        key="question_input"
    )
    
    # Process question
    if user_question:
        with st.spinner("🔍 Searching knowledge base and generating answer..."):
            try:
                # Load RAG chain
                qa_chain, _, _ = load_rag_chain()
                
                # Get answer with sources
                result = qa_chain.invoke({"query": user_question})
                
                answer = result.get("result", "")
                source_docs = result.get("source_documents", [])
                
                # Display answer
                st.markdown("### 💬 Answer")
                st.markdown(answer)
                
                # Display sources
                st.divider()
                with st.expander(f"📖 View Sources ({len(source_docs)} documents)", expanded=True):
                    formatted_sources = format_source_documents(source_docs)
                    st.markdown(formatted_sources)
                
            except Exception as e:
                st.error(f"❌ Error processing question: {str(e)}")
                st.info("💡 Please check your API keys and ensure the knowledge base is initialized")
    
    # Footer
    st.divider()
    st.markdown(
        "<p style='text-align: center; color: gray;'>"
        "Built with LangChain • ChromaDB • Claude 3.5 Sonnet • Voyage AI"
        "</p>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()

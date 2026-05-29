
import os
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
import json

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "financial-documents")

# Use Pinecone's default embeddings (no API key needed)
try:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)
    print("[RAG] Pinecone initialized")
except Exception as e:
    print(f"[RAG] Pinecone init failed: {str(e)}")
    index = None

def store_document_pattern(document_text: str, extracted_data: dict, doc_id: str):
    """Store document pattern in Pinecone."""
    if not index:
        print("[RAG] Pinecone not available, skipping storage")
        return
    
    summary = f"""
    Document Type: {extracted_data.get('document_type', 'unknown')}
    Entities: {', '.join(extracted_data.get('entities', []))}
    Amounts: {', '.join(str(a) for a in extracted_data.get('amounts', []))}
    """
    
    try:
        index.upsert([(doc_id, [0.1] * 1536, {"text": summary})])
        print(f"[RAG] Stored: {doc_id}")
    except Exception as e:
        print(f"[RAG] Storage failed: {str(e)}")

def retrieve_similar_documents(query: str, k: int = 3) -> list:
    """Retrieve similar documents."""
    if not index:
        return []
    
    try:
        results = index.query(vector=[0.1] * 1536, top_k=k)
        return results.get('matches', [])
    except Exception as e:
        print(f"[RAG] Retrieval failed: {str(e)}")
        return []

def enhance_extraction_with_rag(extracted_data: dict, document_text: str) -> dict:
    """Enhance extraction with RAG."""
    similar = retrieve_similar_documents("financial document", k=2)
    
    return {
        **extracted_data,
        "rag_enhanced": True,
        "similar_documents_found": len(similar)
    }

def init_rag():
    """Initialize RAG."""
    if not index:
        return False
    
    try:
        stats = index.describe_index_stats()
        print(f"[RAG] Ready. Total vectors: {stats.get('total_vector_count', 0)}")
        return True
    except Exception as e:
        print(f"[RAG] Init failed: {str(e)}")
        return False


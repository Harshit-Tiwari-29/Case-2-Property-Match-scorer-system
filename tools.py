import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.tools import tool
from typing import List, Dict, Any

# --- TOOL INITIALIZATION ---
# We initialize the DB connection here so the tool has immediate access
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

@tool
def hybrid_search_tool(query: str, max_price: int = 0, min_bedrooms: int = 0) -> List[Dict[str, Any]]:
    """
    Performs a Hybrid Search:
    1. Semantic Similarity Search (Vector) for the 'vibe' and description.
    2. Hard Filtering (Metadata) for strict constraints like Price and Bedrooms.
    
    Args:
        query (str): The semantic search string (e.g., "modern apartment near beach").
        max_price (int): Maximum budget. Use 0 if no limit.
        min_bedrooms (int): Minimum bedrooms required. Use 0 if no limit.
        
    Returns:
        List[Dict]: A list of property dictionaries with metadata and description.
    """
    print(f"🛠️ Tool Call: Hybrid Search | Query: '{query}' | Filters: Max ${max_price}, Min {min_bedrooms} Beds")
    
    # 1. Semantic Search (Fetch more candidates than needed to allow for filtering)
    # We fetch 10 candidates to ensure we don't filter everyone out
    docs = vectorstore.similarity_search(query, k=10)
    
    filtered_results = []
    
    # 2. Metadata Filtering (The "Hard" Logic)
    for doc in docs:
        price = doc.metadata.get("price", 0)
        beds = doc.metadata.get("bedrooms", 0)
        
        # Budget Check
        if max_price > 0 and price > max_price:
            continue # Skip: Too expensive
            
        # Bedroom Check
        if min_bedrooms > 0 and beds < min_bedrooms:
            continue # Skip: Too small
            
        # Add to valid results
        filtered_results.append({
            "id": doc.metadata.get("id"),
            "price": price,
            "bedrooms": beds,
            "bathrooms": doc.metadata.get("bathrooms"),
            "sq_ft": doc.metadata.get("sq_ft"),
            "content": doc.page_content # The description text
        })
    
    # Return top 5 valid matches
    return filtered_results[:5]
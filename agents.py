import os
import json
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from tools import hybrid_search_tool  # <--- Importing your custom tool
from dotenv import load_dotenv  # <--- ADD THIS

load_dotenv()  # <--- ADD THIS (It loads variables from .env)
# --- 1. CONFIGURATION ---
# # Ensure your API key is set in your environment or .env file
# GROQ_API_KEY = os.environ["GROQ_API_KEY"]
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

# --- 2. STATE DEFINITION ---
class AgentState(TypedDict):
    user_input: str                 # Raw input from user
    optimized_query: str            # Semantic search string
    filters: Dict[str, int]         # Structured filters (Budget, Beds)
    retrieved_properties: List[Dict]# Results from the Tool
    final_response: List[Dict]      # Scored & Sorted matches

# --- 3. AGENT: QUERY TRANSFORMATION ---
def query_transform_agent(state: AgentState):
    """
    Role: Transform raw user query into a 'Hybrid Search' query.
    Output: Semantic String + Structured Metadata Filters.
    """
    print(f"🔹 Agent: Transformation | Input: {state['user_input']}")
    
    prompt = f"""
    You are a Query Transformation Agent. 
    Analyze the User Request: "{state['user_input']}"
    
    Return a JSON object with two keys:
    1. "semantic_query": A string optimized for vector similarity search (focus on vibes, location, amenities).
    2. "filters": A dictionary with integer keys "max_price" and "min_bedrooms". Return 0 if not specified.
    
    JSON ONLY.
    """
    response = llm.invoke(prompt)
    try:
        data = json.loads(response.content)
        return {
            "optimized_query": data.get("semantic_query"),
            "filters": data.get("filters", {})
        }
    except:
        # Fallback if JSON parsing fails
        return {"optimized_query": state["user_input"], "filters": {}}

# --- 4. AGENT: RAG TOOL (HYBRID SEARCH) ---
def rag_tool_agent(state: AgentState):
    """
    Role: Orchestrates the search by calling the 'hybrid_search_tool'.
    """
    print(f"🔹 Agent: RAG Tool | Filters: {state['filters']}")
    
    query = state["optimized_query"]
    filters = state["filters"]
    
    # CALL THE TOOL
    # We pass the arguments extracted by the Transformation Agent
    try:
        results = hybrid_search_tool.invoke({
            "query": query,
            "max_price": filters.get("max_price", 0),
            "min_bedrooms": filters.get("min_bedrooms", 0)
        })
    except Exception as e:
        print(f"❌ Tool Error: {e}")
        results = []
    
    return {"retrieved_properties": results}

# --- 5. AGENT: MATCH SCORE CALCULATOR ---
def match_score_agent(state: AgentState):
    """
    Role: Calculate 'Match Score' (0-100) using LLM reasoning.
    """
    print("🔹 Agent: Scoring")
    
    results = []
    properties = state.get("retrieved_properties", [])
    
    if not properties:
        return {"final_response": []}

    prompt_template = """
    Rate the match between User Needs and Property.
    User Needs: {query}
    Property Details: {prop_desc}
    Price: ${price}
    
    Return JSON: {{"score": <int 0-100>, "reason": "<1 short sentence justification>"}}
    """
    
    for prop in properties:
        # Construct a summary string for the LLM to evaluate
        prop_summary = f"{prop.get('content')} | Price: ${prop.get('price')}"
        
        # Invoke LLM
        res = llm.invoke(prompt_template.format(
            query=state["user_input"], 
            prop_desc=prop_summary,
            price=prop.get('price')
        ))
        
        try:
            score_data = json.loads(res.content)
            results.append({
                "id": prop.get("id"),
                "score": score_data["score"],
                "reason": score_data["reason"],
                "details": prop # Pass full details through for the UI
            })
        except:
            continue
            
    # Sort by Score (Highest first)
    results.sort(key=lambda x: x["score"], reverse=True)
    
    return {"final_response": results}

# --- 6. ORCHESTRATION (LANGGRAPH) ---
workflow = StateGraph(AgentState)

# Add Nodes
workflow.add_node("transform", query_transform_agent)
workflow.add_node("rag", rag_tool_agent)
workflow.add_node("score", match_score_agent)

# Define Edges (Linear Logic)
workflow.set_entry_point("transform")
workflow.add_edge("transform", "rag")
workflow.add_edge("rag", "score")
workflow.add_edge("score", END)

# Compile Graph
app_graph = workflow.compile()

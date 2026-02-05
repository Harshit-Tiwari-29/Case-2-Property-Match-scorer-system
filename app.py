import streamlit as st
import pandas as pd
import plotly.express as px
from agents import app_graph # Import the LangGraph application

# Setup
st.set_page_config(layout="wide", page_title="Agent Mira | Multi-Agent RAG")
USER_DB = "data/users.csv"

# --- SIDEBAR: USER SELECTION ---
st.sidebar.title("🤖 Select User Profiles")
st.sidebar.markdown("**System:** Hybrid Search RAG + Multi-Agent")

# Load CLEAN User Data
try:
    # Read CSV with latin1 to handle special characters
    user_df = pd.read_csv(USER_DB, encoding='latin1')
    
    # Create selection list
    # Uses 'Budget' (from original file) instead of 'Budget($)'
    options = user_df.apply(lambda x: f"User {x['User ID']} - Budget: {x['Budget']}", axis=1).tolist()
    selected_user = st.sidebar.selectbox("Select User Profile", options)
    
    # Get User Details based on selection
    user_idx = options.index(selected_user)
    user_data = user_df.iloc[user_idx]
    
    st.sidebar.success(f"Loaded Profile: User {user_data['User ID']}")
    st.sidebar.write(f"**Preferences:** {user_data['Qualitative Description']}")

except Exception as e:
    st.error(f"Could not load users.csv: {e}")
    st.stop()

# --- MAIN PAGE ---
st.title("🏡 Find Your Property Match")

if st.button("🚀 Calculate Matchscore"):
    
    # 1. Construct the Input for the Agents
    # We pass the raw '$500k' string; the Agent is smart enough to parse it.
    agent_input = (
        f"I have a strict budget of {user_data['Budget']}. "
        f"I need minimum {user_data['Bedrooms']} bedrooms. "
        f"My preferences are: {user_data['Qualitative Description']}"
    )
    
    st.info(f"**Agent Input:** {agent_input}")
    
    # 2. Run the LangGraph Pipeline
    with st.spinner("Agents working: Transforming Query -> Hybrid Retrieval -> Scoring..."):
        result = app_graph.invoke({"user_input": agent_input})
        
        # The graph returns 'final_response' as per our agents.py logic
        matches = result.get("final_response", [])

    # 3. Visualization
    if matches:
        st.subheader("Match Analysis")
        
        # DataFrame for Plotly
        df_viz = pd.DataFrame(matches)
        
        # --- CHART UPDATES HERE ---
        # Bar Chart with increased height
        fig = px.bar(
            df_viz, 
            x="score", 
            y="id", 
            orientation='h', 
            color="score", 
            text="score", 
            title="Match Scores",
            labels={"id": "Property ID", "score": "Match Confidence"},
            height=600  # <--- Added Height to make bars bigger
        )
        
        # Optional: Make text inside bars larger too
        fig.update_layout(uniformtext_minsize=12, uniformtext_mode='hide')
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed Cards
        st.subheader("Recommended Properties")
        col1, col2 = st.columns(2)
        
        for i, match in enumerate(matches):
            with col1 if i % 2 == 0 else col2:
                # We use the 'details' dictionary we packed in agents.py
                details = match.get('details', {})
                
                with st.expander(f"Property {match['id']} (Score: {match['score']})", expanded=(i==0)):
                    st.write(f"**Why:** {match['reason']}")
                    # Handle price formatting safely
                    try:
                        price_display = f"${details.get('price', 0):,}"
                    except:
                        price_display = str(details.get('price', 'N/A'))
                        
                    st.write(f"**Price:** {price_display}") 
                    st.write(f"**Specs:** {details.get('bedrooms', '?')} Bed / {details.get('bathrooms', '?')} Bath")
                    st.caption(details.get('content', 'No description available'))
    else:
        st.warning("No suitable matches found within strict constraints.")
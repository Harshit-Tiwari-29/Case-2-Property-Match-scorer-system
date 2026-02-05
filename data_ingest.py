import pandas as pd
import chromadb
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# Load Data
PROP_FILE = "data/properties.csv"  # Ensure your property file is here

def setup_database():
    df = pd.read_csv(PROP_FILE, encoding='latin1')
    
    docs = []
    for _, row in df.iterrows():
        # 1. Prepare Metadata for Keyword/Filtering Search
        # We ensure Price is an integer for mathematical comparison (Hybrid Search requirement)
        try:
            raw_price = str(row['Price']).lower().replace('$', '').replace(',', '')
            price_int = int(float(raw_price.replace('k', '')) * 1000) if 'k' in raw_price else int(float(raw_price))
        except:
            price_int = 0

        metadata = {
            "id": row['Property ID'],
            "price": price_int,
            "bedrooms": row['Bedrooms'],
            "bathrooms": row['Bathrooms'],
            "sq_ft": row['Living Area (sq ft)']
        }

        # 2. Prepare Content for Semantic Search
        # We embed the "Qualitative Description" plus key features
        page_content = f"{row['Qualitative Description']} | Features: {row['Bedrooms']} Bed, {row['Bathrooms']} Bath, {row['Living Area (sq ft)']} sqft."
        
        docs.append(Document(page_content=page_content, metadata=metadata))

    # 3. Create Vector Store
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    print(f"✅ Indexed {len(docs)} properties into Hybrid Vector Store.")

if __name__ == "__main__":
    setup_database()
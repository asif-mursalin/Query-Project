import streamlit as st
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from joblib import load

# Load document embeddings and vectorizer
document_embeddings = np.load("./embeddings.npy")
vectorizer = load('vectorizer.joblib')

# Load documents
with open("documents.txt", "r", encoding="utf-8") as f:
    documents = f.read().split('===DOCUMENT_SEPARATOR===')
documents = [doc.strip() for doc in documents if doc.strip()]

def get_query_embedding(query):
    """Convert query text to embedding using saved vectorizer"""
    query_embedding = vectorizer.transform([query]).toarray()
    return query_embedding[0]

def retrieve_top_k(query_embedding, doc_embeddings, k=5):
    """Retrieve top-k most similar documents using cosine similarity"""
    similarities = cosine_similarity(query_embedding.reshape(1, -1), doc_embeddings)
    top_k_indices = similarities[0].argsort()[-k:][::-1]
    return [(documents[i], similarities[0][i]) for i in top_k_indices]

# Streamlit UI
st.title("Document Search Engine")

# Input query
query = st.text_input("Enter your search query:")

if st.button("Search"):
    if query:
        try:
            # Get query embedding and find similar documents
            query_embedding = get_query_embedding(query)
            results = retrieve_top_k(query_embedding, document_embeddings)
            
            # Display results
            st.write("### Top 5 Most Relevant Documents:")
            for i, (doc, score) in enumerate(results, 1):
                with st.expander(f"Document {i} (Similarity: {score:.3f})"):
                    st.write(doc[:500] + "..." if len(doc) > 500 else doc)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    else:
        st.warning("Please enter a search query.")
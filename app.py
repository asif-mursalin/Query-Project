import streamlit as st
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk

# Download NLTK data if not already downloaded
nltk.download('punkt')

# Load precomputed document embeddings
embeddings = np.load("embeddings.npy")
with open("documents.txt", "r", encoding="utf-8") as f:
    documents = f.read().split('===DOCUMENT_SEPARATOR===')
documents = [doc.strip() for doc in documents if doc.strip()]

# Load Word2Vec model
model = Word2Vec.load("word2vec_model.model")

def get_query_embedding(query, model):
    """Convert query text to embedding using Word2Vec model"""
    # Tokenize and preprocess query
    tokens = word_tokenize(query.lower())
    tokens = [token for token in tokens if token.isalnum()]
    
    # Get embeddings for tokens that exist in the model's vocabulary
    token_embeddings = []
    for token in tokens:
        if token in model.wv:
            token_embeddings.append(model.wv[token])
    
    # If no valid tokens, return zero vector
    if not token_embeddings:
        return np.zeros(model.vector_size)
    
    # Return average of token embeddings
    return np.mean(token_embeddings, axis=0)

def retrieve_top_k(query_embedding, embeddings, k=10):
    """Retrieve top-k most similar documents using cosine similarity."""
    similarities = cosine_similarity(query_embedding.reshape(1, -1), embeddings)
    top_k_indices = similarities[0].argsort()[-k:][::-1]
    return [(documents[i], similarities[0][i]) for i in top_k_indices]

st.title("Information Retrieval using Document Embeddings")

# Input query
query = st.text_input("Enter your query:")

if st.button("Search"):
    if query:
        query_embedding = get_query_embedding(query, model)
        results = retrieve_top_k(query_embedding, embeddings)
        
        # Display results
        st.write("### Top 10 Relevant Documents:")
        for doc, score in results:
            st.write(f"- **Similarity Score:** {score:.4f}")
            st.write(doc[:500] + "..." if len(doc) > 500 else doc)
            st.write("---")
    else:
        st.warning("Please enter a query first.")
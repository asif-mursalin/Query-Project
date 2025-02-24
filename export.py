import nltk
from nltk.corpus import reuters
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from joblib import dump
nltk.download('reuters')
nltk.download('punkt')

# Step 1: Load and process the Reuters Corpus
documents = []
for fileid in reuters.fileids():
    raw_text = reuters.raw(fileid)
    documents.append(raw_text)

print(f"Number of documents: {len(documents)}")

# Step 2: Create document embeddings using TF-IDF
vectorizer = TfidfVectorizer(max_features=100)  # Using 100 features
document_embeddings = vectorizer.fit_transform(documents).toarray()

np.save('document_embeddings.npy', document_embeddings)
print(f"Saved document embeddings with shape: {document_embeddings.shape}")

with open('documents.txt', 'w', encoding='utf-8') as f:
    for doc in documents:
        f.write(doc + "\n===DOCUMENT_SEPARATOR===\n")
print(f"Saved {len(documents)} documents to documents.txt")



dump(vectorizer, 'vectorizer.joblib')
print("Saved vectorizer model")
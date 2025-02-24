import nltk
from nltk.corpus import reuters, stopwords
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Download required NLTK data
nltk.download('reuters')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

# Step 1: Load and process the Reuters Corpus
corpus_sentences = []
document_texts = []  # Store full document texts
for fileid in reuters.fileids():
    raw_text = reuters.raw(fileid)
    # Store the original document text
    document_texts.append(raw_text)
    # Process for Word2Vec
    tokenized_sentence = [word for word in nltk.word_tokenize(raw_text) if word.isalnum() and word]
    corpus_sentences.append(tokenized_sentence)

print(f"Number of sentences in the Reuters corpus: {len(corpus_sentences)}")

# Step 2: Train Word2Vec Model
model = Word2Vec(sentences=corpus_sentences, vector_size=100, window=5, min_count=5, workers=8)

# Step 3: Extract Word Embeddings
words = list(model.wv.index_to_key)
word_vectors = np.array([model.wv[word] for word in words])

# Save embeddings and vocabulary
np.save('embeddings.npy', word_vectors)
print(f"Saved embeddings with shape: {word_vectors.shape}")

# Save documents
with open('documents.txt', 'w', encoding='utf-8') as f:
    for doc in document_texts:
        f.write(doc + "\n===DOCUMENT_SEPARATOR===\n")
print(f"Saved {len(document_texts)} documents to documents.txt")

# Optional: Save vocabulary mapping
with open('vocabulary.txt', 'w', encoding='utf-8') as f:
    for word in words:
        f.write(word + '\n')
print(f"Saved vocabulary with {len(words)} words")

model.save("word2vec_model.model")
print("Saved Word2Vec model as word2vec_model.model")
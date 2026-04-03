from sentence_transformers import SentenceTransformer
import numpy as np

print("RUNNING...")

# Dummy documents
documents = [
    {"page_content": "My name is pawan."},
    {"page_content": "I am currently studying Bachelor of science in Information Technology."},
    {"page_content": "I am interested in artificial intelligence."}
]

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Convert docs to embeddings
texts = [doc["page_content"] for doc in documents]
doc_embeddings = model.encode(texts)

#  USER QUESTION
query = input("\nAsk a question: ")

# Convert question to embedding
query_embedding = model.encode(query)

scores = np.dot(doc_embeddings, query_embedding)

best_index = np.argmax(scores)

print("\nMost relevant answer:")
print(texts[best_index])


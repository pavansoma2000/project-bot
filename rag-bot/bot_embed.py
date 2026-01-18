import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from documents import documents
import pickle

model = SentenceTransformer("all-MiniLM-L6-v2")

texts = [doc["text"] for doc in documents]
embeddings = model.encode(texts)

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

# Save index and docs
faiss.write_index(index, "knowledge.index")
with open("docs.pkl", "wb") as f:
    pickle.dump(documents, f)

print("Knowledge base created successfully")

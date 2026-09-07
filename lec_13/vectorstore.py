import os
import pickle
import numpy as np
import faiss
import sentence_transformers

# ----------------------------
# 1. Load Embedding Model
# ----------------------------
MODEL_NAME = "all-MiniLM-L6-v2"
model = sentence_transformers.SentenceTransformer(MODEL_NAME)
embedding_dim = model.get_embedding_dimension()

# ----------------------------
# 2. Create/Load Persistent FAISS Index
# ----------------------------
INDEX_DIR = "vector_db"
INDEX_PATH = os.path.join(INDEX_DIR, "faiss.index")
METADATA_PATH = os.path.join(INDEX_DIR, "metadata.pkl")

os.makedirs(INDEX_DIR, exist_ok=True)


def _load_index():
    if os.path.exists(INDEX_PATH) and os.path.exists(METADATA_PATH):
        loaded_index = faiss.read_index(INDEX_PATH)
        with open(METADATA_PATH, "rb") as f:
            loaded_metadata = pickle.load(f)
        print(f"Loaded existing FAISS index with {loaded_index.ntotal} vectors.")
    else:
        # IndexFlatIP + normalized embeddings = cosine similarity
        loaded_index = faiss.IndexFlatIP(embedding_dim)
        loaded_metadata = []
    return loaded_index, loaded_metadata


index, metadata_store = _load_index()


def _save_index():
    faiss.write_index(index, INDEX_PATH)
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(metadata_store, f)


# ----------------------------
# 3. Create Embeddings
# ----------------------------
def create_embeddings(chunks):

    if not chunks:
        return np.empty((0, embedding_dim), dtype="float32")

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False
    )
    embeddings = np.asarray(embeddings, dtype="float32")

    print(f"Embeddings created: {len(embeddings)} embeddings")
    return embeddings


# ----------------------------
# 4. Store Chunks
# ----------------------------
def store_chunks(chunks):

    global index, metadata_store

    if not chunks:
        print("No chunks available to store.")
        return

    # Skip chunks with empty/whitespace-only text - encoding them wastes
    # a vector slot and pollutes search results with meaningless matches.
    valid_chunks = [c for c in chunks if c["text"] and c["text"].strip()]
    skipped = len(chunks) - len(valid_chunks)
    if skipped:
        print(f"Skipped {skipped} empty chunk(s).")

    if not valid_chunks:
        print("No non-empty chunks to store.")
        return

    embeddings = create_embeddings(valid_chunks)

    index.add(embeddings)

    for chunk in valid_chunks:
        metadata_store.append({
            "chunk_id": chunk["chunk_id"],
            "text": chunk["text"],
            "source": str(chunk["source"]),
            "page": chunk["page"] if chunk["page"] is not None else 0,
            "chunk_index": chunk["chunk_index"]
        })

    _save_index()

    print(f"Stored {len(valid_chunks)} chunks in FAISS index (total in index: {index.ntotal}).")


# ----------------------------
# 5. Semantic Search
# ----------------------------
def search_chunks(question, top_k=3):

    if index.ntotal == 0:
        print("Vector database is empty.")
        return None

    query_embedding = model.encode(
        [question],
        normalize_embeddings=True
    ).astype("float32")

    k = min(top_k, index.ntotal)
    scores, indices = index.search(query_embedding, k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        entry = metadata_store[idx]
        results.append({
            "chunk_id": entry["chunk_id"],
            "text": entry["text"],
            "source": entry["source"],
            "page": entry["page"],
            "chunk_index": entry["chunk_index"],
            "similarity_score": round(float(score), 4)
        })

    return results
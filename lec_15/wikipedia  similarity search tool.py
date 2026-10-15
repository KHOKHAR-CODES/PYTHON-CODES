import requests
from bs4 import BeautifulSoup
import json
import os
import re
from sentence_transformers import SentenceTransformer

JSON_FILE = "wikipedia_chunks.json"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # small, fast, good for demo


# -------------------------
# JSON helpers
# -------------------------

def load_chunks(path: str):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        return []
    return data


def save_chunks(path: str, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# -------------------------
# Text cleaning
# -------------------------

def clean_text(text: str) -> str:
    """
    Clean raw scraped text:
      - Remove citation/reference markers like [1], [note 1], etc.
      - Remove commas and some extra punctuation
      - Normalize whitespace
    """
    # Remove bracketed references
    text = re.sub(r"\[[^\]]*\]", "", text)

    # Remove commas and some other punctuation if you want very clean text
    text = text.replace(",", "")
    text = text.replace(";", "")
    text = text.replace('"', "")
    text = text.replace("'", "")

    # Optionally keep only letters, numbers, spaces and basic sentence punctuation
    # text = re.sub(r"[^A-Za-z0-9.\s]", "", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


# -------------------------
# Chunking
# -------------------------

def create_chunks(
    documents,
    chunk_size=10,
    overlap=2,
    min_chunk_words=1
):
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    step = chunk_size - overlap
    chunks = []
    chunk_counter = 1

    for doc in documents:
        text = doc["content"]
        words = text.split()
        start = 0
        index = 0

        while start < len(words):
            end = start + chunk_size
            chunk_words = words[start:end]

            if len(chunk_words) < min_chunk_words:
                break

            chunk_text = " ".join(chunk_words)
            chunks.append({
                "chunk_id": chunk_counter,
                "text": chunk_text,
                "source": doc["source"],
                "page": doc["page"],
                "chunk_index": index
            })
            chunk_counter += 1
            index += 1

            if end >= len(words):
                break

            start += step

    return chunks


# -------------------------
# Wikipedia fetch
# -------------------------

def fetch_wikipedia_document(topic: str):
    topic_clean = topic.replace(" ", "_")
    url = f"https://en.wikipedia.org/wiki/{topic_clean}"

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0 Safari/537.36"
        )
    }

    try:
        resp = requests.get(url, headers=headers, timeout=10)
    except requests.RequestException as e:
        print(f"Request error: {e}")
        return None

    print(f"\nStatus Code: {resp.status_code}")

    if resp.status_code != 200:
        print("Page not found or inaccessible.")
        return None

    print("Status: Success - Page loaded successfully")

    soup = BeautifulSoup(resp.text, "html.parser")

    title_tag = soup.find("h1")
    title = title_tag.get_text(" ", strip=True) if title_tag else "Unknown title"
    print(f"Title: {title}")

    content_div = soup.find("div", class_="mw-parser-output")
    if not content_div:
        print("Content section not found.")
        return None

    elements = content_div.find_all(["p", "li"])
    paragraphs = []
    for el in elements:
        text = el.get_text(" ", strip=True)
        text
        if text:
            paragraphs.append(text)

    if not paragraphs:
        print("No textual content found.")
        return None

    raw_text = "\n\n".join(paragraphs)
    cleaned_text = clean_text(raw_text)

    return {
        "content": cleaned_text,
        "source": url,
        "page": None,
        "title": title
    }


# -------------------------
# Embeddings & similarity
# -------------------------

def load_model():
    return SentenceTransformer(EMBEDDING_MODEL)


def compute_embeddings(model, texts):
    # texts: list of strings
    return model.encode(texts, convert_to_numpy=True, show_progress_bar=False)


def cosine_similarity(a, b):
    # a, b: 1D numpy arrays
    import numpy as np
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def search_chunks(query: str, chunks, model, top_k: int = 5):
    """
    Return top_k most similar chunks for a query string.
    Each result: (score, chunk_dict)
    """
    import numpy as np

    if not chunks:
        return []

    query_emb = compute_embeddings(model, [query])[0]

    chunk_texts = [c["text"] for c in chunks]
    chunk_embs = compute_embeddings(model, chunk_texts)

    scores = []
    for i, emb in enumerate(chunk_embs):
        sim = cosine_similarity(query_emb, emb)
        scores.append((sim, chunks[i]))

    # Sort by similarity descending
    scores.sort(key=lambda x: x[0], reverse=True)
    return scores[:top_k]


# -------------------------
# Main flow
# -------------------------

def main():
    topic = input("Enter Wikipedia topic: ").strip()
    if not topic:
        print("No topic provided. Exiting.")
        return

    doc = fetch_wikipedia_document(topic)
    if not doc:
        print("Failed to fetch document.")
        return

    documents = [doc]

    # Create chunks
    chunks = create_chunks(documents, chunk_size=10, overlap=2)

    if not chunks:
        print("No chunks created.")
        return

    # Load existing chunks and append
    existing = load_chunks(JSON_FILE)

    if existing:
        max_id = max(c.get("chunk_id", 0) for c in existing)
        offset = max_id
        for i, c in enumerate(chunks):
            c["chunk_id"] = offset + i + 1
        all_chunks = existing + chunks
    else:
        all_chunks = chunks

    save_chunks(JSON_FILE, all_chunks)

    print(f"\nCreated {len(chunks)} chunks for this topic.")
    print(f"Total chunks in file: {len(all_chunks)}")
    print(f"Saved to: {JSON_FILE}")

    # ---- Embeddings & similarity demo ----

    print("\nLoading embedding model (first time may take a while)...")
    model = load_model()

    print("\nExample similarity search:")
    query = input("Enter a search query about the topic: ").strip()
    if query:
        results = search_chunks(query, all_chunks, model, top_k=5)
        print(f"\nTop {len(results)} chunks for query: '{query}'\n")
        for i, (score, chunk) in enumerate(results, start=1):
            print(f"{i}. Score: {score:.4f}")
            print("Text:", chunk["text"][:200], "...")
            print("Source:", chunk["source"])
            print("-" * 60)

    # Optional: direct similarity between two sentences
    print("\nDirect similarity between two sentences:")
    s1 = input("Sentence 1: ").strip()
    s2 = input("Sentence 2: ").strip()
    if s1 and s2:
        e1, e2 = compute_embeddings(model, [s1, s2])
        sim = cosine_similarity(e1, e2)
        print(f"Similarity score: {sim:.4f} (0 = unrelated, 1 = very similar)")


if __name__ == "__main__":
    main()
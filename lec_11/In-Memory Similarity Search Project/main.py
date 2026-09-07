import json
from pathlib import Path
from sentence_transformers import SentenceTransformer


# Load the embedding model once when the program starts.
model = SentenceTransformer("all-MiniLM-L6-v2")


def cosine_similarity(vector_a, vector_b):
    """Calculate similarity between two embedding vectors."""

    # Multiply matching vector values and add them together.
    dot_product = sum(a * b for a, b in zip(vector_a, vector_b))

    # Calculate the size (magnitude) of each vector.
    magnitude_a = sum(a * a for a in vector_a) ** 0.5
    magnitude_b = sum(b * b for b in vector_b) ** 0.5

    # Return the cosine similarity score.
    return dot_product / (magnitude_a * magnitude_b)


def main():
    # Load records from the JSON knowledge base.
    records = json.loads(
        Path("knowledge_base.json").read_text(encoding="utf-8")
    )

    # Extract the content text from every knowledge-base record.
    texts = [record["content"] for record in records]

    # Create an embedding vector for every stored text.
    document_embeddings = model.encode(texts)

    # Get the user's search query.
    query = input("Enter your query: ")

    # Create an embedding vector for the user query.
    query_embedding = model.encode(query)

    results = []

    # Compare the query embedding with each document embedding.
    for record, embedding in zip(records, document_embeddings):
        score = cosine_similarity(
            query_embedding.tolist(),
            embedding.tolist()
        )

        # Store each record together with its similarity score.
        results.append((record, score))

    # Sort results from highest similarity score to lowest.
    results.sort(key=lambda item: item[1], reverse=True)

    print("\nTop 3 Results:")

    # Display only the top three matching records.
    for record, score in results[:3]:
        print(f"\nTitle: {record['title']}")
        print(f"Score: {score:.4f}")
        print(f"Content: {record['content']}")


# Run the program only when this file is executed directly.
if __name__ == "__main__":
    main()
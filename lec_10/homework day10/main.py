"""
Load labeled sentences from sentences.json, generate sentence embeddings
for each one using the all-MiniLM-L6-v2 model, and print the results.
"""

import json
from pathlib import Path

from sentence_transformers import SentenceTransformer

SENTENCES_PATH = Path(__file__).parent / "sentences.json"
MODEL_NAME = "all-MiniLM-L6-v2"


def load_records(path: Path) -> list:
    """Load sentence records from a JSON file."""
    with open(path, "r", encoding="utf-8") as records:
        return json.load(records)


def main() :
    records = load_records(SENTENCES_PATH)
    texts = [record["text"] for record in records]
    topics= sorted(set(record["topic"] for record in records))

    print(f"Loaded {len(records)} records covering {len(topics)} topics: {', '.join(topics)}")
    print(f"Loading model '{MODEL_NAME}'...")

    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(texts)
    
    print(f"\nComplete embeddings shape: {embeddings.shape}\n")
    print("-" * 70)

    for record, embedding in zip(records, embeddings):
        dimension = embedding.shape[0]
        first_five = ", ".join(f"{value:.4f}" for value in embedding[:5])

        print(f"Text:       {record['text']}")
        print(f"Topic:      {record['topic']}")
        print(f"Dimension:  {dimension}")
        print(f"First 5:    [{first_five}]")
        print("-" * 70)


if __name__ == "__main__":
    main()

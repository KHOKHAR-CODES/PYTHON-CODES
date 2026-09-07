import json
from pathlib import Path
from sentence_transformers import SentenceTransformer

# Load python
 records from the JSON file
file_path = Path("knowledge_base.json")
PYTHON = json.loads(file_path.read_text())

# Create document embeddings once
model = SentenceTransformer("all-MiniLM-L6-v2")
python_texts = [python["text"] for python in PYTHON]
python_vectors = model.encode(python_texts)


def get_cosine_score(query_vector, python_vector):
    dot_product = sum(a * b for a, b in zip(query_vector, python_vector))
    query_size = sum(a * a for a in query_vector) ** 0.5
    python_size = sum(b * b for b in python_vector) ** 0.5
    return dot_product / (query_size * python_size)


def find_best_matches(question):
    query_vector = model.encode(question)
    matches = []

    for python, python_vector in zip(PYTHON, python_vectors):
        score = get_cosine_score(query_vector, python_vector)
        matches.append({"python": python, "score": score})

    matches.sort(key=lambda match: match["score"], reverse=True)
    return matches[:3]


while True:
    question = input("\nAsk python relavant questions (or type quit):").strip()
    if question.lower() == "quit":
        print("   Good Bye   ")
        break
    if question == "":
        print("Please enter a question.")
        continue

    for number, match in enumerate(find_best_matches(question), start=1):
        python = match["python"]
        print(f"\n{number}. {python['text']}")
        print(f"Source: {python['source']}")
        print(f"Score: {match['score']:.3f}") 
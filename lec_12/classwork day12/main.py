import json
import chromadb

# Load data from JSON file
with open("knowledge_base.json", "r", encoding="utf-8") as file:
    data = json.load(file)

# Create/load persistent ChromaDB database
client = chromadb.PersistentClient(path="chroma_db")

collection = client.get_or_create_collection(
    name="company_policies"
)

# Add data only if the collection is empty
if collection.count() == 0:
    ids = []
    documents = []

    for item in data:
        ids.append(str(item["id"]))
        documents.append(item["text"])

    collection.add(
        ids=ids,
        documents=documents
    )

    print("Knowledge base added successfully.")
else:
    print("Existing ChromaDB database loaded.")

# Start semantic search
print("\nSimple semantic search engine")
print("Type 'exit' to close the program.")

while True:
    query = input("\nAsk a question: ").strip()

    # Exit program
    if query.lower() == "exit" or "quit":
        print("Program closed.")
        break

    # Empty input
    if not query:
        print("Please enter a question.")
        continue

    # Search ChromaDB
    results = collection.query(
        query_texts=[query],
        n_results=2
    )

    # Display results
    print("\nTop results:")
    print("_" * 50)

    for number, result in enumerate(results["documents"][0], start=1):
        print(f"{number}. {result}")
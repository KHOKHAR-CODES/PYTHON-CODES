# List of sentences covering Python, AI, RAG, embeddings, documents, and vector databases
sentences = [
    "Python is the primary programming language used for modern AI development.",
    "Artificial Intelligence models rely heavily on high-quality training datasets.",
    "Retrieval-Augmented Generation (RAG) enhances LLM responses using external data.",
    "Embeddings convert text documents into numerical vector representations.",
    "Vector databases store high-dimensional embeddings for fast similarity searches.",
    "A document can be chunked into smaller sections to improve retrieval accuracy.",
    "Python libraries like LangChain simplify building RAG applications.",
    "Cosine similarity measures how close two embeddings are in vector space.",
    "Using vector databases helps scale retrieval across millions of documents.",
    "RAG prevents AI hallucination by anchoring answers in grounded documents."
]

print("--- AI & RAG Sentence Search Engine ---")
print("Enter a keyword to search, or type 'exit' to quit.\n")

# Main program loop using a while statement
while True:
    user_input = input("Enter search keyword: ")

    # Normalize user input by stripping extra whitespace and converting to lowercase
    keyword = user_input.strip().lower()

    # Reject empty inputs and restart the loop
    if not keyword:
        print("Input cannot be empty. Please try again.\n")
        continue

    # Exit condition using a break statement
    if keyword == "exit":
        print("Goodbye!")
        break

    # List to store matching sentences
    matches = []

    # Iterate through all sentences using a for loop
    for sentence in sentences:
        # Perform a case-insensitive match check
        if keyword in sentence.lower():
            matches.append(sentence)

    # Display search results or "No result found" message
    if matches:
        print(f"\nFound {len(matches)} match(es):")
        for index, match in enumerate(matches, start=1):
            print(f"{index}. {match}")
    else:
        print("No result found.")

    print("\n" + "-" * 40)
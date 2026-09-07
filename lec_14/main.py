import json
import os
import builtins
from document_loader import load_document
from text_cleaner import clean_text
from chunk_creator import create_chunks
from vectorstore import store_chunks, search_chunks

# -----------------------------------
# Input/Output file paths
# -----------------------------------
def i():
    document_options = {
    "1": r"D:\python\Usama-Rasheed\day13\classwork\documents dir\employee_training_karachi.txt",
    "2": r"D:\python\Usama-Rasheed\day13\classwork\documents dir\employee_training_lahore.docx",
    "3": r"D:\python\Usama-Rasheed\day13\classwork\documents dir\Unit01_Digital_Logic_Chapter1.pdf",
}
    choice = builtins.input("Enter the document number (1-3): ").strip()
    if choice not in document_options:
        print("Invalid choice. Please select a valid document number (1-3).")
        return i()

document_options, choice = i()
INPUT_FILE = document_options[choice]
OUTPUT_DIR = "output"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "chunks.json")

#------------------------------------------
# Load Document
#------------------------------------------
documents = load_document(INPUT_FILE)
print(f"Loaded pages/documents: {len(documents)}")

#------------------------------------------
# Clean Text
#------------------------------------------
for doc in documents:
    doc["content"] = clean_text(
        doc["content"]
    )

#------------------------------------------
# Create Chunks
#------------------------------------------
chunks = create_chunks(
    documents,
    chunk_size=120,
    overlap=20
)
print(f"Total chunks created: {len(chunks)}")

#------------------------------------------
# Save chunks to Json
#------------------------------------------
os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(
        chunks,
        f,
        indent=4,
        ensure_ascii=False
    )
print("RAG chunks created successfully!")
print(f"Saved at: {OUTPUT_FILE}")

#------------------------------------------
# Create Embeddings & Store in Vector DB
# (this step was missing before, which is why
# no embeddings were ever generated)
#------------------------------------------
store_chunks(chunks)

#------------------------------------------
# Example semantic search with similarity scores
#------------------------------------------
query = "What is this document about?"
results = search_chunks(query, top_k=3)

if results:
    print(f"\nTop {len(results)} results for query: {query!r}\n")
    for rank, r in enumerate(results, start=1):
        print(f"{rank}. [score={r['similarity_score']}] "
              f"(source={r['source']}, page={r['page']}, chunk={r['chunk_id']})")
        print(f"   {r['text'][:150]}...\n")
# In-Memory Similarity Search

This project searches a small JSON knowledge base using text embeddings.

## How It Works

1. Load records from `knowledge_base.json`
2. Create embeddings for all record content
3. Ask the user for a query
4. Create an embedding for the query
5. Calculate cosine similarity scores
6. Sort results from highest to lowest
7. Display the top 3 matches

## Files

```text
In-Memory Similarity Search Project\
├── main.py
├── knowledge_base.json
├── requirements.txt
└── README.md
```

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

## Example Query

```text
Enter your query: What are embeddings?
```

The program displays the three knowledge-base records most similar to the query.

## Requirements

```text
sentence-transformers
```

This project uses in-memory search and does not use a vector database.
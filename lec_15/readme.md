# Wikipedia Chunker & Semantic Search

A small command-line tool that scrapes a Wikipedia article, cleans and splits it into overlapping text chunks, stores them in a local JSON file, and lets you run semantic (embedding-based) search over those chunks using `sentence-transformers`.

## Features

- **Scrapes** any Wikipedia article by topic name (`requests` + `BeautifulSoup`)
- **Cleans** the text: strips `[citation]` references, commas, semicolons, and quote characters, and normalizes whitespace
- **Chunks** the article into fixed-size, overlapping word windows
- **Persists** chunks to `wikipedia_chunks.json`, appending new topics to existing data without overwriting
- **Embeds & searches**: encodes chunks with the `all-MiniLM-L6-v2` sentence-transformer model and ranks them by cosine similarity to a query
- **Direct sentence comparison**: compare the similarity of any two sentences you type in

## Requirements

- Python 3.8+
- Packages:

  ```bash
  pip install requests beautifulsoup4 sentence-transformers
  ```

## Usage

Run the script from the command line:

```bash
python main.py
```

You'll be prompted for:

1. **Wikipedia topic** — e.g. `Python (programming language)`. Spaces are converted to underscores to build the Wikipedia URL.
2. **A search query** — the tool embeds all stored chunks (old + new) and prints the top 5 most similar ones with their similarity scores.
3. **Two sentences** (optional) — get a direct cosine-similarity score between them.

### Example session

Enter Wikipedia topic: Python (programming language)

Status Code: 200
Status: Success - Page loaded successfully
Title: Python (programming language)

Created 42 chunks for this topic.
Total chunks in file: 42
Saved to: wikipedia_chunks.json

Loading embedding model (first time may take a while)...

Example similarity search:
Enter a search query about the topic: Who created Python?

Top 5 chunks for query: 'Who created Python?'

1. Score: 0.6123
   Text: Python was created by Guido van Rossum and first released in 1991 ...
   Source: <https://en.wikipedia.org/wiki/Python_(programming_language)>

---

...

## Output format

Chunks are stored in `wikipedia_chunks.json` as a list of objects:

```json
[
  {
    "chunk_id": 1,
    "text": "Python may refer to: Pythonidae a family of nonvenomous snakes",
    "source": "https://en.wikipedia.org/wiki/python",
    "page": null,
    "chunk_index": 0
  }
]
```

Running the script again with a new topic **appends** new chunks to the existing file, continuing the `chunk_id` numbering from the current maximum.

## Configuration

These are set as constants near the top of the script:

| Constant          | Default                 | Description                                     |
| ----------------- | ----------------------- | ----------------------------------------------- |
| `JSON_FILE`       | `wikipedia_chunks.json` | Path to the chunk storage file                  |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2`      | Sentence-transformers model used for embeddings |

Chunking behavior can be tuned via `create_chunks()` parameters:

| Parameter         | Default | Description                                       |
| ----------------- | ------- | ------------------------------------------------- |
| `chunk_size`      | `10`    | Number of words per chunk                         |
| `overlap`         | `2`     | Number of words shared between consecutive chunks |
| `min_chunk_words` | `1`     | Minimum words required to keep a trailing chunk   |

## Notes & known limitations

- `JSON_FILE` is currently a **relative path**, so the file is written to whatever directory the script is _run from_, not necessarily the folder the script lives in. If you need it to always save next to the script regardless of your working directory, anchor it with:

  ```python
  BASE_DIR = os.path.dirname(os.path.abspath(__file__))
  JSON_FILE = os.path.join(BASE_DIR, "wikipedia_chunks.json")
  ```

- `clean_text()` strips standard ASCII commas, semicolons, and quotes. Non-ASCII comma-like characters (e.g. `、`, `،`) are not removed by default.
- The scraper pulls all `<p>` and `<li>` text from the article body, so it will include list items (e.g. disambiguation entries, "See also" lists) alongside prose paragraphs.
- No rate limiting or caching is implemented — each run makes a fresh HTTP request to Wikipedia.

## License

Use freely for personal or educational purposes.

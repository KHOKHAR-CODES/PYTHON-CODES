# Python Similarity Search

This small program searches Python learning notes using semantic similarity. It embeds the notes and the user question with the `all-MiniLM-L6-v2` model, calculates cosine similarity, and displays the three closest matches.

## Project files

- `main.py` — the interactive search program.
- `knowledge_base.json` — nine Python-related records. Every record has `id`, `text`, and `source`.
- `requirement.txt` — required Python package.

## Install

Use Python 3.4 or later. In the project folder, run:

```bash
pip install -r requirement.txt
```

## Run

Keep `main.py` and `knowledge_base.json` in the same folder, then run:

```bash
python main.py
```

Enter a question when prompted. Type `quit` to close the program. If Enter is pressed without a question, the program prints `Please enter a question.`

## Example test questions

Test the program with at least three different questions:

```text
How do I add an item to a list?
How can I handle errors in Python?
How do I read a text file?
```

You can also try:

```text
What is a dictionary?
How do I install a Python package?
```

## How the code works

1. Loads the records from `knowledge_base.json`.
2. Creates embeddings for all record text using `all-MiniLM-L6-v2`.
3. Converts the user's question into an embedding.
4. Calculates cosine similarity with a Python function.
5. Sorts matches from highest to lowest score.
6. Displays the top three matches with the text, source, and score rounded to three decimal places.

## Imports in the program

```python
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
```

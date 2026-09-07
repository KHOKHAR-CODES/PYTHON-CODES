# Sentence Embedding Demo

A small project that loads a set of labeled sentences from `sentences.json`,
generates a sentence embedding for each one using the `all-MiniLM-L6-v2`
model (via [sentence-transformers](https://www.sbert.net/)), and prints the
results to the console.

## Project structure

```
.
├── main.py            # loads the sentences, generates embeddings, prints results
├── sentences.json      # 10 sentence records across 4 topics
├── requirements.txt    # Python dependencies
└── README.md            # this file
```

`sentences.json` contains 10 records spanning 4 topics: **technology**,
**nature**, **food**, and **sports**.

## Setup

1. **Create and activate a virtual environment** (recommended)

   ```bash
   python3 -m venv venv
   source venv/bin/activate      # on Windows: venv\Scripts\activate
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

   This installs `sentence-transformers`, which in turn pulls in `torch`
   and other dependencies. The first install may take a few minutes.

## Run

```bash
python main.py
```

On first run, the script downloads the `all-MiniLM-L6-v2` model
(~90 MB) from the Hugging Face Hub, so an **internet connection is
required the first time**. The model is cached locally afterward
(typically under `~/.cache/torch/sentence_transformers/`), so
subsequent runs are fast and work offline.

## Expected output

The script prints:

1. The complete embeddings array shape, e.g.

   ```
   Complete embeddings shape: (10, 384)
   ```

2. For every record: the sentence text, its topic, the embedding
   dimension (384 for this model), and the first five values of its
   embedding vector, e.g.

   ```
   Text:       Artificial intelligence is transforming how software is designed and deployed.
   Topic:      technology
   Dimension:  384
   First 5:    [0.0123, -0.0456, 0.0789, -0.0234, 0.0567]
   ```

## Requirements

- Python 3.9+
- Internet access on first run (to download the model)

## Customizing

To use your own data, edit `sentences.json`. Each record needs at
least a `text` field and a `topic` field:

```json
{ "id": 11, "topic": "history", "text": "Your sentence here." }
```

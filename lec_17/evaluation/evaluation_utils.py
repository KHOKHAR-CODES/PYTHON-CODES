"""Shared helpers for both manual and RAGAS evaluation scripts."""

import json
import os
from typing import List

import pandas as pd


def load_eval_dataset(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def populate_predictions(dataset: List[dict], pipeline) -> List[dict]:
    """Run the RAG pipeline over each question in the dataset and fill
    in the generated answer and retrieved contexts, so the dataset is
    ready for scoring against its ground_truth field."""
    for item in dataset:
        result = pipeline.answer(item["question"])
        item["answer"] = result["answer"]
        item["contexts"] = [c["text"] for c in result["contexts"]]
    return dataset


def save_results_csv(rows: List[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"[evaluation_utils] Saved results to {path}")

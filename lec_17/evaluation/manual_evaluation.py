"""Manual evaluation: runs the RAG pipeline over evaluation_data.json,
then walks a human reviewer through each question/answer/context so
they can score Faithfulness, Answer Relevance, Context Precision, and
Context Recall by hand (0.0-1.0). Results are saved to results/.

Run:
    python -m evaluation.manual_evaluation
"""

import os
from dotenv import load_dotenv

load_dotenv()

from src.rag_pipeline import RAGPipeline
from evaluation.evaluation_utils import load_eval_dataset, populate_predictions, save_results_csv

EVAL_DATA_PATH = os.getenv("EVAL_DATA_PATH", "evaluation/evaluation_data.json")
RESULTS_PATH = os.getenv("MANUAL_RESULTS_PATH", "results/evaluation_report.csv")

METRICS = ["faithfulness", "answer_relevance", "context_precision", "context_recall"]


def score_prompt(metric: str) -> float:
    while True:
        raw = input(f"  {metric} (0.0 - 1.0): ").strip()
        try:
            value = float(raw)
            if 0.0 <= value <= 1.0:
                return value
        except ValueError:
            pass
        print("  Please enter a number between 0.0 and 1.0.")


def main():
    dataset = load_eval_dataset(EVAL_DATA_PATH)
    pipeline = RAGPipeline()
    dataset = populate_predictions(dataset, pipeline)

    rows = []
    for item in dataset:
        print("\n" + "=" * 60)
        print(f"Question: {item['question']}")
        print(f"Generated Answer: {item['answer']}")
        print(f"Ground Truth: {item['ground_truth']}")
        print(f"Retrieved Contexts: {len(item['contexts'])} chunk(s)")
        for i, ctx in enumerate(item["contexts"], start=1):
            preview = ctx[:200] + ("..." if len(ctx) > 200 else "")
            print(f"  [{i}] {preview}")

        print("\nRate this answer manually:")
        scores = {m: score_prompt(m) for m in METRICS}

        row = {"question": item["question"], "answer": item["answer"], "ground_truth": item["ground_truth"]}
        row.update(scores)
        rows.append(row)

    save_results_csv(rows, RESULTS_PATH)


if __name__ == "__main__":
    main()

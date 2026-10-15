"""Automated evaluation using RAGAS, scoring Faithfulness, Answer
Relevance, Context Precision, and Context Recall.

RAGAS needs a "judge" LLM and embedding model to score the answers.
This uses Hugging Face for both:
- Judge LLM:   a Hugging Face Inference API model (needs HF_TOKEN),
               via langchain_huggingface.HuggingFaceEndpoint.
- Embeddings:  the same local sentence-transformers model used for
               retrieval, via langchain_huggingface.HuggingFaceEmbeddings.

Run:
    python -m evaluation.ragas_evaluation

Note: RAGAS's API has changed across versions. If `evaluate(...)`
below doesn't match your installed version's signature, check
`pip show ragas` and the RAGAS docs for the current metric/LLM
wiring — the overall flow (build dataset -> evaluate -> save CSV)
stays the same.
"""

import os
from dotenv import load_dotenv

load_dotenv()

from src.rag_pipeline import RAGPipeline
from evaluation.evaluation_utils import load_eval_dataset, populate_predictions, save_results_csv

EVAL_DATA_PATH = os.getenv("EVAL_DATA_PATH", "evaluation/evaluation_data.json")
RESULTS_PATH = os.getenv("RAGAS_RESULTS_PATH", "results/ragas_results.csv")
RAGAS_JUDGE_MODEL = os.getenv("RAGAS_JUDGE_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")


def build_ragas_dataset(dataset: list):
    from datasets import Dataset

    return Dataset.from_list([
        {
            "question": item["question"],
            "answer": item["answer"],
            "contexts": item["contexts"],
            "ground_truth": item["ground_truth"],
        }
        for item in dataset
    ])


def main():
    dataset = load_eval_dataset(EVAL_DATA_PATH)
    pipeline = RAGPipeline()
    dataset = populate_predictions(dataset, pipeline)

    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
    from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings as LCHFEmbeddings

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise EnvironmentError(
            "HF_TOKEN is required for RAGAS evaluation (used as the judge LLM "
            "via the Hugging Face Inference API). Add it to your .env file."
        )

    judge_llm = HuggingFaceEndpoint(
        repo_id=RAGAS_JUDGE_MODEL,
        huggingfacehub_api_token=hf_token,
        max_new_tokens=512,
    )
    judge_embeddings = LCHFEmbeddings(
        model_name=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    )

    ragas_dataset = build_ragas_dataset(dataset)

    result = evaluate(
        ragas_dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=judge_llm,
        embeddings=judge_embeddings,
    )

    df = result.to_pandas()
    print(df)
    save_results_csv(df.to_dict(orient="records"), RESULTS_PATH)


if __name__ == "__main__":
    main()

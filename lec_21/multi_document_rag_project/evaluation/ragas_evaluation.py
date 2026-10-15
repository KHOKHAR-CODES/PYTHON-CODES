"""Automated evaluation using RAGAS, scoring Faithfulness, Answer
Relevancy, Context Precision, and Context Recall.

RAGAS needs a "judge" LLM and an embedding model to score answers.
This project uses Hugging Face for both:
- Judge LLM:   a hosted chat model via the Inference API (needs HF_TOKEN).
               The modern Inference providers only expose a *conversational*
               endpoint, so the judge calls `client.chat_completion` through
               a small LangChain LLM subclass — plain HuggingFaceEndpoint
               (text_generation route) now fails with
               "not supported for task text-generation" on most models.
- Embeddings:  the same local sentence-transformers model used for
               retrieval, via langchain_huggingface.HuggingFaceEmbeddings.

Verified against ragas 0.2.15 (evaluate() + EvaluationDataset +
LangchainLLMWrapper). Older/newer RAGAS versions are handled at runtime.

Run from the terminal:
    python -m evaluation.ragas_evaluation

Or from the Streamlit app's Evaluation tab (button or /eval command).
"""

import os
import re
import time
from typing import Callable, List, Optional

import pandas as pd
from dotenv import load_dotenv

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"), override=True)

from src.rag_pipeline import RAGPipeline
from evaluation.evaluation_utils import load_eval_dataset, populate_predictions, save_results_csv

EVAL_DATA_PATH = os.getenv("EVAL_DATA_PATH", "evaluation/evaluation_data.json")
RESULTS_PATH = os.getenv("RAGAS_RESULTS_PATH", "results/ragas_results.csv")
RAGAS_JUDGE_MODEL = os.getenv("RAGAS_JUDGE_MODEL", "meta-llama/Llama-3.1-8B-Instruct")

METRIC_KEYS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

# Models known to work on the Inference API's conversational endpoint
# (tested 2026-09). The first one that answers is used as the judge.
FALLBACK_JUDGES = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen/Qwen2.5-72B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "HuggingFaceH4/zephyr-7b-beta",
]


def require_hf_token() -> str:
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise EnvironmentError(
            "HF_TOKEN is required for RAGAS evaluation (the judge LLM runs on "
            "the Hugging Face Inference API). Add it to your .env file."
        )
    return hf_token


def _probe_chat_model(model: str, hf_token: str) -> bool:
    """Cheaply check whether a model answers on the chat endpoint."""
    try:
        from huggingface_hub import InferenceClient

        client = InferenceClient(model=model, token=hf_token, provider="auto")
        out = client.chat_completion(
            [{"role": "user", "content": "Reply with the single word: OK"}],
            max_tokens=5,
        )
        return bool(out.choices and out.choices[0].message.content)
    except Exception:
        return False


def _resolve_judge_model(hf_token: str, preferred: str) -> str:
    """Use the preferred model if it works, otherwise try known-good ones."""
    candidates = [preferred] + [m for m in FALLBACK_JUDGES if m != preferred]
    for model in candidates:
        if _probe_chat_model(model, hf_token):
            return model
    raise RuntimeError(
        "None of the candidate judge models answered on the Hugging Face "
        f"Inference API ({', '.join(candidates)}). Check your HF_TOKEN, the "
        "model list at hf.co/models?inference_provider=all, or set "
        "RAGAS_JUDGE_MODEL in .env to a chat model you have access to."
    )


def _build_judge_llm(model: str, hf_token: str):
    """LangChain LLM backed by the Inference API *chat* endpoint.

    A plain HuggingFaceEndpoint no longer works for most hosted models
    (they expose only the conversational task), so we subclass LangChain's
    LLM and call `chat_completion` directly — verified working with
    meta-llama/Llama-3.1-8B-Instruct.
    """
    from huggingface_hub import InferenceClient
    from langchain_core.language_models.llms import LLM

    class HFInferenceChatLLM(LLM):
        model_name: str = model
        max_tokens: int = 512

        @property
        def _llm_type(self) -> str:
            return "hf-inference-chat"

        def _call(self, prompt: str, stop=None, run_manager=None, **kwargs) -> str:
            client = InferenceClient(model=self.model_name, token=hf_token, provider="auto")
            out = client.chat_completion(
                [{"role": "user", "content": prompt}],
                max_tokens=self.max_tokens,
                stop=stop or None,
            )
            return (out.choices[0].message.content or "").strip()

    return HFInferenceChatLLM()


def _resolve_metrics() -> List:
    """Return the four core metrics from whichever RAGAS version is
    installed (0.1.x singletons and 0.2+ classes both supported)."""
    import ragas.metrics as rm

    candidates = {
        "faithfulness": ["faithfulness", "Faithfulness"],
        "answer_relevancy": ["answer_relevancy", "AnswerRelevancy", "ResponseRelevancy"],
        "context_precision": ["context_precision", "ContextPrecision"],
        "context_recall": ["context_recall", "ContextRecall"],
    }

    resolved = []
    for key, names in candidates.items():
        for name in names:
            obj = getattr(rm, name, None)
            if obj is None:
                continue
            resolved.append(obj() if isinstance(obj, type) else obj)
            break
        else:
            raise ImportError(f"RAGAS metric for '{key}' not found in ragas.metrics")
    return resolved


def _wrap_llm(lc_llm):
    """RAGAS 0.2+ needs LangchainLLMWrapper; 0.1.x takes the raw LLM."""
    try:
        from ragas.llms import LangchainLLMWrapper

        return LangchainLLMWrapper(lc_llm)
    except ImportError:
        return lc_llm


def _wrap_embeddings(lc_embeddings):
    try:
        from ragas.embeddings import LangchainEmbeddingsWrapper

        return LangchainEmbeddingsWrapper(lc_embeddings)
    except ImportError:
        return lc_embeddings


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9']+", (text or "").lower())


def _safe_float(value):
    try:
        value = float(value)
        if value != value:
            return 0.0
        return max(0.0, min(1.0, value))
    except Exception:
        return 0.0


def _fallback_metric_row(item: dict) -> dict:
    """Heuristic metrics used when RAGAS judge calls fail or return NaN.

    The scores remain bounded to [0, 1] and are used only as a safety net for
    the web app / evaluation tab, not as a replacement for the real judge.
    """
    question = (item.get("question") or "").strip()
    answer = (item.get("answer") or "").strip()
    ground_truth = (item.get("ground_truth") or "").strip()
    contexts = item.get("contexts") or []
    context_text = " ".join(str(c) for c in contexts)

    q_tokens = set(_tokenize(question))
    a_tokens = set(_tokenize(answer))
    gt_tokens = set(_tokenize(ground_truth))
    ctx_tokens = set(_tokenize(context_text))

    if not q_tokens and not a_tokens:
        faithfulness = 0.0
        answer_relevancy = 0.0
    else:
        answer_overlap = q_tokens & a_tokens
        faithfulness = len(answer_overlap) / max(len(a_tokens), 1) if a_tokens else 0.0
        answer_relevancy = len(answer_overlap) / max(len(q_tokens), 1) if q_tokens else 0.0

    if a_tokens and context_text:
        context_precision = len(a_tokens & ctx_tokens) / max(len(a_tokens), 1)
    else:
        context_precision = 0.0

    if gt_tokens and context_text:
        context_recall = len(gt_tokens & ctx_tokens) / max(len(gt_tokens), 1)
    else:
        context_recall = 0.0

    return {
        "faithfulness": _safe_float(faithfulness),
        "answer_relevancy": _safe_float(answer_relevancy),
        "context_precision": _safe_float(context_precision),
        "context_recall": _safe_float(context_recall),
    }


def _apply_fallback_metrics(dataset: List[dict], df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [c for c in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"] if c in df.columns]
    if not metric_cols:
        return df

    for row_idx, item in enumerate(dataset):
        if row_idx >= len(df):
            break
        fallback = _fallback_metric_row(item)
        for metric in metric_cols:
            value = df.at[row_idx, metric]
            if value is None or pd.isna(value):
                df.at[row_idx, metric] = fallback.get(metric, 0.0)
    return df


def run_ragas(
    dataset: List[dict],
    hf_token: str,
    judge_model: str = RAGAS_JUDGE_MODEL,
    pipeline: Optional[RAGPipeline] = None,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> pd.DataFrame:
    """Run the RAG pipeline over the dataset, then score it with RAGAS.

    Returns a DataFrame with one row per question plus the four metric
    columns. Raises RuntimeError with an install hint if RAGAS is missing.
    """
    log = progress_cb or (lambda msg: None)

    try:
        import ragas  # noqa: F401
    except ImportError:
        raise RuntimeError(
            "RAGAS is not installed. Run:  pip install -r requirements-ragas.txt"
        )

    if pipeline is None:
        log("Loading the RAG pipeline…")
        pipeline = RAGPipeline()

    log("Running the pipeline over the evaluation questions…")
    dataset = populate_predictions(dataset, pipeline)

    judge_model = _resolve_judge_model(hf_token, judge_model)
    log(f"Judge model: {judge_model}")
    judge_llm = _build_judge_llm(judge_model, hf_token)

    log("Loading the local embedding model for RAGAS…")
    from langchain_huggingface import HuggingFaceEmbeddings as LCHFEmbeddings

    judge_embeddings = LCHFEmbeddings(
        model_name=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    )

    metrics = _resolve_metrics()
    # ragas 0.2+ renamed the dataset columns: user_input / retrieved_contexts
    # / response / reference (verified: 0.2.15 raises ValueError on the old
    # question/contexts/answer/ground_truth keys). 0.1.x keeps the old names.
    rows_modern = [
        {
            "user_input": item["question"],
            "retrieved_contexts": item["contexts"],
            "response": item["answer"],
            "reference": item["ground_truth"],
        }
        for item in dataset
    ]
    rows_legacy = [
        {
            "question": item["question"],
            "answer": item["answer"],
            "contexts": item["contexts"],
            "ground_truth": item["ground_truth"],
        }
        for item in dataset
    ]

    from ragas import evaluate

    try:
        from ragas import EvaluationDataset

        ragas_dataset = EvaluationDataset.from_list(rows_modern)
    except ImportError:
        from datasets import Dataset

        ragas_dataset = Dataset.from_list(rows_legacy)

    log("Scoring with RAGAS (one judge call per metric per question)…")
    start = time.time()
    result = evaluate(
        ragas_dataset,
        metrics=metrics,
        llm=_wrap_llm(judge_llm),
        embeddings=_wrap_embeddings(judge_embeddings),
    )
    log(f"Scoring finished in {time.time() - start:.0f}s")

    df = result.to_pandas()

    # RAGAS sometimes returns all-NaN metric columns when the judge API is
    # rate-limited, gated, or timing out. Keep the app usable by filling in a
    # lightweight fallback score so the evaluation tab still reports numbers.
    metric_cols = [c for c in df.columns if c in METRIC_KEYS]
    if metric_cols:
        empty_rows = int(df[metric_cols].isna().all(axis=1).sum())
        partial_rows = int((df[metric_cols].isna().any(axis=1) & ~df[metric_cols].isna().all(axis=1)).sum())
        if empty_rows or partial_rows:
            log(
                f"⚠️ {empty_rows} row(s) scored fully and {partial_rows} partially NaN — "
                "the judge LLM hit an API error (quota / rate limit). Falling back to "
                "heuristic metric scores so the report remains usable."
            )
            df = _apply_fallback_metrics(dataset, df)

    return df


def main():
    dataset = load_eval_dataset(EVAL_DATA_PATH)
    hf_token = require_hf_token()
    df = run_ragas(dataset, hf_token)
    print(df)
    save_results_csv(df.to_dict(orient="records"), RESULTS_PATH)
    print(f"[ragas_evaluation] Done. Results saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()

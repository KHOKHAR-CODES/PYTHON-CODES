"""Handles the connection to a Hugging Face Large Language Model.

Two providers are supported, chosen via the LLM_PROVIDER env var:
- "local":         runs a model on this machine with `transformers`
                    (default: google/flan-t5-base — small and CPU-friendly,
                    needs no token, no internet access after the first
                    download).
- "inference_api":  calls the Hugging Face Inference API using an HF_TOKEN,
                    for larger hosted models without downloading them.

The inference_api path uses `provider="auto"` so huggingface_hub can route
the request to any Inference Provider (HF's own serverless tier, Together,
Fireworks, Groq, etc.) that currently hosts the requested model, and it
automatically falls back to a short list of known-good chat models if the
configured one is gated, deprecated, or not deployed anywhere — the same
resilience pattern used for the RAGAS judge model in
evaluation/ragas_evaluation.py.
"""

import os

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "local")
LLM_MODEL = os.getenv("LLM_MODEL", "google/flan-t5-base")
HF_TOKEN = os.getenv("HF_TOKEN")

# Models that reliably answer on the HF Inference Providers' chat endpoint
# without requiring a gated-model access request. Used as a fallback when
# the configured LLM_MODEL doesn't work (wrong name, gated, not deployed,
# or temporarily unavailable).
FALLBACK_CHAT_MODELS = [
    "Qwen/Qwen2.5-7B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "HuggingFaceH4/zephyr-7b-beta",
]


class HFLLM:
    def __init__(self, provider: str = LLM_PROVIDER, model_name: str = LLM_MODEL):
        self.provider = provider
        self.model_name = model_name

        if self.provider == "local":
            from transformers import AutoConfig, logging

            logging.set_verbosity_error()
            config = AutoConfig.from_pretrained(model_name)
            if config.is_encoder_decoder:
                from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
                self.pipe = None
            else:
                from transformers import pipeline

                self.pipe = pipeline("text-generation", model=model_name)
        elif self.provider == "inference_api":
            if not HF_TOKEN:
                raise EnvironmentError(
                    "HF_TOKEN is not set. Add it to your .env file to use the "
                    "Hugging Face Inference API (LLM_PROVIDER=inference_api), "
                    "or set LLM_PROVIDER=local to run a model on this machine "
                    "instead (no token required)."
                )
            self.client, self.model_name = self._resolve_client(model_name, HF_TOKEN)
        else:
            raise ValueError(f"Unknown LLM_PROVIDER: '{self.provider}' (use 'local' or 'inference_api')")

    @staticmethod
    def _probe(client_model: str, token: str) -> bool:
        """Cheaply check whether a model answers on the chat endpoint."""
        try:
            from huggingface_hub import InferenceClient

            probe_client = InferenceClient(model=client_model, token=token, provider="auto")
            out = probe_client.chat_completion(
                [{"role": "user", "content": "Reply with the single word: OK"}],
                max_tokens=5,
            )
            return bool(out.choices and out.choices[0].message.content)
        except Exception:
            return False

    @classmethod
    def _resolve_client(cls, preferred_model: str, token: str):
        """Pick the first model (preferred, then fallbacks) that actually
        answers on the Inference API, and return a ready client for it.

        Without this, a gated, renamed, or undeployed model just raises an
        opaque error the first time someone asks a question — this mirrors
        the fallback already used for the RAGAS judge model."""
        from huggingface_hub import InferenceClient

        candidates = [preferred_model] + [m for m in FALLBACK_CHAT_MODELS if m != preferred_model]
        for candidate in candidates:
            if cls._probe(candidate, token):
                return InferenceClient(model=candidate, token=token, provider="auto"), candidate

        raise RuntimeError(
            "None of the candidate chat models answered on the Hugging Face "
            f"Inference API ({', '.join(candidates)}). This usually means "
            "HF_TOKEN is invalid/expired, your account has no Inference "
            "Providers credit, or the models require accepting a license at "
            "huggingface.co first. You can also set LLM_PROVIDER=local in "
            ".env to skip the API entirely."
        )

    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        if self.provider == "local":
            if self.pipe is not None:
                output = self.pipe(prompt, max_new_tokens=max_new_tokens, do_sample=False)
                return output[0]["generated_text"].strip()

            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
            return self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

        # Modern Inference providers only expose the *conversational* task;
        # text_generation fails with "not supported for task text-generation".
        try:
            response = self.client.chat_completion(
                [{"role": "user", "content": prompt}],
                max_tokens=max_new_tokens,
            )
            return (response.choices[0].message.content or "").strip()
        except Exception as error:
            raise RuntimeError(
                f"Hugging Face Inference API call failed for model "
                f"'{self.model_name}': {error}. Check your HF_TOKEN, billing "
                "at https://hf.co/settings/billing, or switch LLM_PROVIDER=local "
                "in .env."
            ) from error

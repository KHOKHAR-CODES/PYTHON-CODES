"""Handles the connection to a Hugging Face Large Language Model.

Two providers are supported, chosen via the LLM_PROVIDER env var:
- "local":         runs a model on this machine with `transformers`
                    (default: google/flan-t5-base — small and CPU-friendly).
- "inference_api":  calls the Hugging Face Inference API using an HF_TOKEN,
                    for larger hosted models without downloading them.
"""

import os

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "local")
LLM_MODEL = os.getenv("LLM_MODEL", "google/flan-t5-base")
HF_TOKEN = os.getenv("HF_TOKEN")


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
            from huggingface_hub import InferenceClient
            if not HF_TOKEN:
                raise EnvironmentError(
                    "HF_TOKEN is not set. Add it to your .env file to use the "
                    "Hugging Face Inference API (LLM_PROVIDER=inference_api)."
                )
            self.client = InferenceClient(model=model_name, token=HF_TOKEN)
        else:
            raise ValueError(f"Unknown LLM_PROVIDER: '{self.provider}' (use 'local' or 'inference_api')")

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

        response = self.client.text_generation(prompt, max_new_tokens=max_new_tokens)
        return response.strip() if isinstance(response, str) else str(response)

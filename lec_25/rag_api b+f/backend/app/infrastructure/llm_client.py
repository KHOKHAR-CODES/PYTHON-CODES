from __future__ import annotations

import time
from abc import ABC, abstractmethod


from app.core.exceptions import LLMError, LLMTimeoutError


class LLMClient(ABC):
    model_name: str

    @abstractmethod
    def generate_answer(self, system_prompt: str, user_prompt: str) -> tuple[str, int]:
        raise NotImplementedError

    def health(self) -> str:
        return 'ready'


class OpenAICompatibleLLMClient(LLMClient):
    """Hugging Face Inference Providers via the OpenAI-compatible SDK."""

    def __init__(
        self,
        api_key: str,
        model: str = 'openai/gpt-oss-20b',
        timeout: float = 30.0,
        max_output_tokens: int = 700,
        retry_attempts: int = 1,
        retry_backoff: float = 0.25,
    ):
        if not api_key:
            raise LLMError('HF_TOKEN is required for Hugging Face inference.')
        try:
            from openai import OpenAI
            from openai import APIConnectionError, APITimeoutError, APIStatusError, RateLimitError
        except ImportError as exc:
            raise LLMError('The openai package is required for Hugging Face inference.') from exc
        self._APIConnectionError = APIConnectionError
        self._APITimeoutError = APITimeoutError
        self._APIStatusError = APIStatusError
        self._RateLimitError = RateLimitError
        self.model_name = model
        self.max_output_tokens = max_output_tokens
        self.retry_attempts = retry_attempts
        self.retry_backoff = retry_backoff
        self.client = OpenAI(
            base_url='https://openrouter.ai/api/v1',
            api_key=api_key,
            timeout=timeout,
            max_retries=0,
        )

    def generate_answer(self, system_prompt: str, user_prompt: str) -> tuple[str, int]:
        started = time.perf_counter()
        for attempt in range(self.retry_attempts + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {'role': 'system', 'content': system_prompt},
                        {'role': 'user', 'content': user_prompt},
                    ],
                    temperature=0,
                    max_tokens=self.max_output_tokens,
                )
                answer = response.choices[0].message.content if response.choices else None
                if not isinstance(answer, str) or not answer.strip():
                    raise LLMError('Language model returned an empty answer.')
                return answer.strip(), int((time.perf_counter() - started) * 1000)
            except LLMError:
                raise
            except self._APITimeoutError as exc:
                if attempt < self.retry_attempts:
                    time.sleep(self.retry_backoff * (2 ** attempt))
                    continue
                raise LLMTimeoutError() from exc
            except (self._RateLimitError, self._APIConnectionError) as exc:
                if attempt < self.retry_attempts:
                    time.sleep(self.retry_backoff * (2 ** attempt))
                    continue
                raise LLMError('Hugging Face inference provider is temporarily unavailable.') from exc
            except self._APIStatusError as exc:
                if exc.status_code in {408, 429} or exc.status_code >= 500:
                    if attempt < self.retry_attempts:
                        time.sleep(self.retry_backoff * (2 ** attempt))
                        continue
                    raise LLMError(f'Hugging Face inference provider returned HTTP {exc.status_code}.') from exc
                raise LLMError(f'Hugging Face inference provider rejected the request ({exc.status_code}).') from exc
            except (ValueError, IndexError, AttributeError, TypeError) as exc:
                raise LLMError('Hugging Face inference provider returned an invalid response.') from exc

    def health(self) -> str:
        return 'ready'

    def close(self) -> None:
        self.client.close()


class TestLLMClient(LLMClient):
    """Deterministic local LLM substitute used only by the automated test environment."""
    model_name = "test-llm"

    def generate_answer(self, system_prompt: str, user_prompt: str) -> tuple[str, int]:
        tail = user_prompt.split("=== CURRENT QUESTION ===", 1)[-1]
        question = next((line.strip() for line in tail.splitlines() if line.strip()), "test question")
        return f"Retrieved context answer for: {question}", 0

import json
import os
from typing import Any

from dotenv import load_dotenv
from huggingface_hub import InferenceClient


SYSTEM_MESSAGE = {
    "role": "system",
    "content": (
        "You are a friendly, patient assistant for beginners. Explain ideas in "
        "plain language, use small examples when helpful, and say when you are "
        "not sure."
    ),
}
MAX_RECENT_MESSAGES = 8


def require_settings() -> tuple[str, str]:
    """Read the required values after dotenv has loaded them."""
    token = os.getenv("HF_TOKEN")
    model_id = os.getenv("MODEL_ID")
    if not token or not model_id:
        raise RuntimeError(
            "Missing HF_TOKEN or MODEL_ID. Copy .env.example to .env and fill in both values."
        )
    return token, model_id


def parse_json_answer(text: str) -> Any | None:
    """ It Return JSON When Ask."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def ask(client: InferenceClient, history: list[dict[str, str]], question: str, temperature: float) -> str:
    """Send the question with only the most recent chat context."""
    messages = [SYSTEM_MESSAGE, *history, {"role": "user", "content": question}]
    response = client.chat_completion(
        messages=messages,
        max_tokens=500,
        temperature=temperature,
    )
    answer = response.choices[0].message.content or "I did not receive a text response."
    history.extend(
        [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
    )
    del history[:-MAX_RECENT_MESSAGES]
    return answer


def print_help() -> None:
    print("\nCommands: /help, /clear, /temperature 0.1-1.0, /exit")
    print("Tip: ask normally, or request JSON with a prompt such as:")
    print('  Return ONLY valid JSON: {"topic":"Python", "difficulty":"beginner"}\n')


def main() -> None:
    load_dotenv()
    try:
        token, model_id = require_settings()
    except RuntimeError as error:
        print(f"Configuration error: {error}")
        return

    client = InferenceClient(model=model_id, token=token)
    history: list[dict[str, str]] = []
    temperature = 0.7

    print(f"Beginner Chat is ready (model: {model_id}). Type /help for commands.")
    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not question:
            print("Please enter a question (or /exit to quit).")
            continue
        if question.lower() in {"/exit", "exit", "quit"}:
            print("Goodbye!")
            break
        if question.lower() == "/help":
            print_help()
            continue
        if question.lower() == "/clear":
            history.clear()
            print("Conversation history cleared.")
            continue
        if question.lower().startswith("/temperature"):
            parts = question.split(maxsplit=1)
            try:
                temperature = float(parts[1])
                if not 0.1 <= temperature <= 1.0:
                    raise ValueError
                print(f"Temperature set to {temperature:.1f}.")
            except (IndexError, ValueError):
                print("Use /temperature followed by a number from 0.1 to 1.0.")
            continue

        try:
            answer = ask(client, history, question, temperature)
        except Exception as error:
            print(f"Request failed: {error}")
            continue

        print(f"Assistant: {answer}")
        json_answer = parse_json_answer(answer)
        if json_answer is not None:
            print("Parsed JSON:")
            print(json.dumps(json_answer, indent=2))


if __name__ == "__main__":
    main()
import os
import json
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# --------------------------------------------------
# 1. Load environment variables
# --------------------------------------------------

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_ID = os.getenv("HF_MODEL_ID")


# --------------------------------------------------
# 2. Validate token and model
# --------------------------------------------------

if not HF_TOKEN:
    raise ValueError(

        
        "HF_TOKEN is missing. Please add it to your .env file."
    )

if not MODEL_ID:
    raise ValueError(
        "HF_MODEL_ID is missing. Please add it to your .env file."
    )


# --------------------------------------------------
# 3. Create Hugging Face client
# --------------------------------------------------

client = InferenceClient(
    model=MODEL_ID,
    token=HF_TOKEN
)


# --------------------------------------------------
# Helper function for API calls
# --------------------------------------------------

def ask_model(messages, max_tokens=300):
    """
    Send messages to the Hugging Face model
    and return the assistant response.
    """

    try:
        response = client.chat_completion(
            messages=messages,
            max_tokens=max_tokens
        )

        return response.choices[0].message.content

    except Exception as error:
        print("\nAPI Error:")
        print(error)

        return None


# --------------------------------------------------
# 4. First API call
# --------------------------------------------------

def first_api_call():

    print("\n" + "=" * 50)
    print("FIRST API CALL")
    print("=" * 50)

    messages = [
        {
            "role": "user",
            "content": (
                "Explain Artificial Intelligence "
                "in one simple sentence."
            )
        }
    ]

    answer = ask_model(messages)

    if answer:
        print("\nAI Response:")
        print(answer)


# --------------------------------------------------
# 5. Weak vs improved prompt
# --------------------------------------------------

def compare_prompts():

    print("\n" + "=" * 50)
    print("PROMPT COMPARISON")
    print("=" * 50)

    weak_prompt = "Tell me about Python."

    improved_prompt = """
You are a programming teacher.

Explain Python to a complete beginner.

Include:
1. What Python is
2. Three common uses of Python
3. One simple Python code example

Use simple language.
Keep the answer under 150 words.
"""

    prompts = [
        ("Weak Prompt", weak_prompt),
        ("Improved Prompt", improved_prompt)
    ]

    for title, prompt in prompts:

        print("\n" + "-" * 50)
        print(title)
        print("-" * 50)

        answer = ask_model(
            [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        if answer:
            print(answer)


# --------------------------------------------------
# 6. Request and parse JSON
# --------------------------------------------------

def json_example():

    print("\n" + "=" * 50)
    print("JSON OUTPUT EXAMPLE")
    print("=" * 50)

    prompt = """
Give information about the Python programming language.

Return ONLY valid JSON.

Use exactly this structure:

{
    "language": "",
    "difficulty": "",
    "uses": []
}

Do not use markdown.
Do not use ```json.
Do not write anything before or after the JSON.
"""

    answer = ask_model(
        [
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens=200
    )

    if not answer:
        return

    print("\nRaw AI Response:")
    print(answer)

    try:
        data = json.loads(answer)

        print("\nParsed JSON:")
        print(data)

        print("\nLanguage:")
        print(data["language"])

        print("\nDifficulty:")
        print(data["difficulty"])

        print("\nUses:")

        for use in data["uses"]:
            print("-", use)

    except json.JSONDecodeError:
        print("\nError: The model did not return valid JSON.")


# --------------------------------------------------
# 7. Chatbot
# --------------------------------------------------

def chatbot():

    system_message = {
        "role": "system",
        "content": (
            "You are a helpful AI assistant. "
            "Give simple, clear and concise answers."
        )
    }

    history = [system_message]

    print("\n" + "=" * 50)
    print("HUGGING FACE CHATBOT")
    print("=" * 50)

    print("\nCommands:")
    print("exit    -> Close chatbot")
    print("clear   -> Clear conversation history")
    print("history -> Show conversation history")

    while True:

        user_input = input("\nYou: ").strip()

        # -----------------------------
        # Empty input
        # -----------------------------

        if not user_input:
            print("Chatbot: Please enter a message.")
            continue

        # -----------------------------
        # Exit command
        # -----------------------------

        if user_input.lower() == "exit":
            print("Chatbot: Goodbye!")
            break

        # -----------------------------
        # Clear command
        # -----------------------------

        if user_input.lower() == "clear":

            history = [system_message]

            print("Chatbot: Conversation history cleared.")
            continue

        # -----------------------------
        # History command
        # -----------------------------

        if user_input.lower() == "history":

            print("\nConversation History:")
            print("-" * 40)

            if len(history) == 1:
                print("No conversation yet.")

            else:

                for message in history[1:]:

                    role = message["role"].capitalize()
                    content = message["content"]

                    print(f"{role}: {content}")

            continue

        # -----------------------------
        # Add user message
        # -----------------------------

        history.append(
            {
                "role": "user",
                "content": user_input
            }
        )

        # -----------------------------
        # Send history to model
        # -----------------------------

        answer = ask_model(
            history,
            max_tokens=300
        )

        if answer:

            print(f"\nChatbot: {answer}")

            history.append(
                {
                    "role": "assistant",
                    "content": answer
                }
            )

        else:
            # Remove failed user message
            history.pop()


# --------------------------------------------------
# Main menu
# --------------------------------------------------

def main():

    while True:

        print("\n" + "=" * 50)
        print("HUGGING FACE LLM PROJECT")
        print("=" * 50)

        print("1. First API Call")
        print("2. Compare Weak and Improved Prompts")
        print("3. JSON Output Example")
        print("4. Start Chatbot")
        print("5. Exit")

        choice = input("\nSelect option: ").strip()

        if choice == "1":
            first_api_call()

        elif choice == "2":
            compare_prompts()

        elif choice == "3":
            json_example()

        elif choice == "4":
            chatbot()

        elif choice == "5":
            print("\nProgram closed.")
            break

        else:
            print("\nInvalid option. Please select 1-5.")


# --------------------------------------------------
# Run program
# --------------------------------------------------

if __name__ == "__main__":
    main()
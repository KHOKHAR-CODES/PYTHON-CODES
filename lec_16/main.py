import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import InferenceClient

from langchain_core.output_parsers import StrOutputParser
from prompts import study_prompt

def create_client():
    project_folder = Path(__file__).resolve().parent
    env_path = project_folder / ".env"
    load_dotenv(dotenv_path=env_path, override=True)

    token = os.getenv("HF_TOKEN")
    model_id = os.getenv("MODEL_ID")

    if not token:
        raise ValueError("HF_TOKEN is not set in the .env file.")
    if not model_id:
        raise RuntimeError("MODEL_ID is not set in the .env file.")

    print("loaded model: ", model_id)
    client = InferenceClient(token=token, model=model_id)
    return client



def main():
    try:
        client = create_client()
        print("\nAI study Helper")
        print("Type 'exit' to quit the program.\n")
        while True:
            topic = input("Topic: ").strip()

            if topic.lower() == "exit":
                print("Goodbye!")
                break

            if not topic:
                print("Please enter a valid topic.")
                continue
            level = input("Level (beginner/intermediate): ").strip()   

            if not level:
                level = "beginner"

            messages = study_prompt.format_messages(topic = topic, level = level)
            hf_messages = []

            for message in messages:
                if message.type == "system":
                    role = "system"
                else:
                    role = "user"

                hf_messages.append({"role": role, "content": message.content})

            print("\nGenerating explaination...\n")
            response = client.chat_completion(messages = hf_messages, max_tokens = 500, temperature = 0.3)
            answer = response.choices[0].message.content

            print("Raw response object: ")
            print(response)

            print("\nPlain text: ")
            print(answer)
            print("\n" + "-" * 50 + "\n")

    except Exception as error:
        print("\nThe program could not run.")
        print("\nTechinal details: ", error)



if __name__ == "__main__":
    main()
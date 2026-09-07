import os
from google import genai

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise EnvironmentError(
        "GEMINI_API_KEY environment variable is not set. "
        "Create a .env file (see .env.example) or set it in your shell."
    )

client = genai.Client(api_key=GEMINI_API_KEY)


def generate_answer(question, results, model_name="gemini-3.6-flash"):

    if not results:
        return "No relevant context found to answer the question."

    context = "\n\n".join(
        f"[Source: {r['source']}, page: {r['page']}]\n{r['text']}"
        for r in results
    )

    prompt = f"""Answer the question using only the context below.
If the answer isn't in the context, say you don't know.

Context:
{context}

Question: {question}
"""

    response = client.models.generate_content(
        model=model_name,
        contents=prompt
    )

    return response.text
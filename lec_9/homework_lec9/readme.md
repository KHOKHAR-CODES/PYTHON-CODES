**Beginner Chat CLI**

1. Create a virtual environment (optional): python -m venv .venv
2. Install packages: "pip install -r requirements.txt'
3. Copy ".env.example" to ".env", then add your Hugging Face token and model ID.
4. Run: "python app.py"

Useful commands: "/clear", "/temperature 0.1-1.0", and "/exit".

The chat retains only the last four turns, so context stays recent and requests remain small.

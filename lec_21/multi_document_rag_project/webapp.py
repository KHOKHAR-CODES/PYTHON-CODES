"""Browser interface for the multi-document RAG system."""

import contextlib
import io
import os

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

from src.rag_pipeline import RAGPipeline


app = Flask(__name__)
pipeline = None
pipeline_error = None


def get_pipeline():
    global pipeline, pipeline_error
    if pipeline is None and pipeline_error is None:
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                pipeline = RAGPipeline()
        except Exception as error:
            pipeline_error = str(error)
    return pipeline


@app.get("/")
def index():
    return render_template("index.html", ready=get_pipeline() is not None)


@app.post("/api/ask")
def ask():
    data = request.get_json(silent=True) or {}
    question = str(data.get("question", "")).strip()

    if not question:
        return jsonify({"error": "Ask a question to search your documents."}), 400

    rag_pipeline = get_pipeline()
    if rag_pipeline is None:
        return jsonify({"error": pipeline_error or "The knowledge desk could not start."}), 503

    try:
        return jsonify(rag_pipeline.answer(question))
    except Exception as error:
        return jsonify({"error": str(error)}), 500


if __name__ == "__main__":
    host = os.getenv("WEB_HOST", "127.0.0.1")
    port = int(os.getenv("WEB_PORT", "5000"))
    app.run(host=host, port=port, debug=False)
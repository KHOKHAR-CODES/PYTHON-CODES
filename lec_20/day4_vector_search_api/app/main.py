"""FastAPI application entry point for the Day 4 vector search API."""

from fastapi import FastAPI

from app.routes import router

app = FastAPI(
    title="Day 4 Vector Search API",
    description="Retrieval stage of a RAG pipeline: index files and search them by similarity.",
    version="1.0.0",
)

app.include_router(router)


@app.get("/")
def root() -> dict:
    """Basic health-check / welcome endpoint."""
    return {
        "message": "Day 4 Vector Search API is running. Visit /docs for interactive API docs.",
    }

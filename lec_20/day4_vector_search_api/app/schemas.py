"""Pydantic models (schemas) used by the API request/response bodies."""

from typing import List

from pydantic import BaseModel, Field


class IndexRequest(BaseModel):
    """Request body for POST /index-file."""

    file_path: str = Field(..., description="Absolute or relative path to a .txt, .pdf, or .docx file.")


class IndexResponse(BaseModel):
    """Response body returned after a file has been indexed."""

    file_name: str
    total_chunks: int
    vectors_stored: int
    status: str


class SearchRequest(BaseModel):
    """Request body for POST /search."""

    question: str = Field(..., description="Natural language question to search for.")
    top_k: int = Field(default=3, description="Number of top matching chunks to return.")


class SearchResult(BaseModel):
    """A single retrieved chunk and the source file it came from."""

    chunk: str
    source: str


class SearchResponse(BaseModel):
    """Response body returned for a search query."""

    question: str
    results: List[SearchResult]

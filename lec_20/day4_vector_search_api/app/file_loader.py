"""Load raw text content from TXT, PDF, and DOCX files."""

import os

from pypdf import PdfReader
from docx import Document


class FileLoaderError(Exception):
    """Raised when a file cannot be located or its type is unsupported."""


SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".docx"}


def load_file(file_path: str) -> str:
    """Load and return the raw text content of a TXT, PDF, or DOCX file.

    Args:
        file_path: Path to the file to load.

    Returns:
        The extracted raw text content of the file.

    Raises:
        FileLoaderError: If the file does not exist or has an unsupported extension,
            or if the file cannot be read/parsed.
    """
    if not os.path.exists(file_path):
        raise FileLoaderError(f"File not found: '{file_path}'.")

    if not os.path.isfile(file_path):
        raise FileLoaderError(f"Path is not a file: '{file_path}'.")

    _, extension = os.path.splitext(file_path)
    extension = extension.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise FileLoaderError(
            f"Unsupported file extension '{extension}'. "
            f"Supported extensions are: {', '.join(sorted(SUPPORTED_EXTENSIONS))}."
        )

    try:
        if extension == ".txt":
            return _load_txt(file_path)
        if extension == ".pdf":
            return _load_pdf(file_path)
        return _load_docx(file_path)
    except FileLoaderError:
        raise
    except Exception as exc:  # noqa: BLE001 - wrap any parser-specific error clearly
        raise FileLoaderError(f"Failed to read file '{file_path}': {exc}") from exc


def _load_txt(file_path: str) -> str:
    """Read raw text from a .txt file."""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def _load_pdf(file_path: str) -> str:
    """Extract text from a .pdf file using pypdf."""
    reader = PdfReader(file_path)
    pages_text = []
    for page in reader.pages:
        extracted = page.extract_text() or ""
        pages_text.append(extracted)
    return "\n".join(pages_text)


def _load_docx(file_path: str) -> str:
    """Extract text from a .docx file using python-docx."""
    document = Document(file_path)
    paragraphs = [paragraph.text for paragraph in document.paragraphs]
    return "\n".join(paragraphs)

import os
from pypdf import PdfReader
from docx import Document


def load_txt(file_path):
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    for encoding in ("utf-8-sig", "utf-16", "cp1252", "latin-1"):
        try:
            with open(file_path, "r", encoding=encoding) as file:
                text = file.read()
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError(f"Unable to decode text file: {file_path}")

    return [{
        "content": text,
        "source": file_path,
        "page": None
    }]


def load_pdf(file_path):

    reader = PdfReader(file_path)

    documents = []

    for page_number, page in enumerate(reader.pages):

        text = page.extract_text()

        documents.append({
            "content": text,
            "source": file_path,
            "page": page_number + 1
        })

    return documents

def load_docx(file_path):

    doc = Document(file_path)

    text = "\n".join(
        paragraph.text 
        for paragraph in doc.paragraphs
    )
    return [{
        "content": text,
        "source": file_path,
        "page": None
    }]

def load_document(file_path):

    extension = os.path.splitext(file_path)[1].lower()

    if extension == ".txt":
        return load_txt(file_path)

    elif extension == ".pdf":
        return load_pdf(file_path)

    elif extension == ".docx":
        return load_docx(file_path)
    else:
        raise Exception(
            "Unsupported file type"
        )
from langchain_huggingface import HuggingFaceEmbeddings


def get_embedder(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=model_name)


def embed_text(text: str, embedder: HuggingFaceEmbeddings = None) -> list:
    if embedder is None:
        embedder = get_embedder()
    return embedder.embed_query(text)


def embed_documents(texts: list, embedder: HuggingFaceEmbeddings = None) -> list:
    if embedder is None:
        embedder = get_embedder()
    return embedder.embed_documents(texts)


if __name__ == "__main__":
    sample_output = "This is some text produced by another part of the pipeline."

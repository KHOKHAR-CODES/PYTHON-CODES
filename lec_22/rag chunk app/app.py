import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

st.title("Document Chunking APP")
uploaded_file = st.file_uploader(
    "upload a pdf file",
    type=["pdf"]
)

if uploaded_file is not None:
    reader = PdfReader(uploaded_file)
    full_text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            full_text += page_text + "\n"

    st.success("text extracted successfully")
    st.subheader("extracted text")
    st.text_area(
        "document text",
        full_text,
        height=250
    )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=2
    )
    chunks = text_splitter.split_text(full_text)

    st.subheader("chunks")

    st.write("total chunks:", len(chunks))
    for i, chunk in enumerate(chunks):
        with st.expander(f"chunk {i + 1}"):
            st.write(chunk)
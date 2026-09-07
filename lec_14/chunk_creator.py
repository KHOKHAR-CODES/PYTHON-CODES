def create_chunks(
        documents,
        chunk_size=10,
        overlap=2
):
    chunks=[]
    chunk_counter=1
    for doc in documents:
        text = doc["content"]
        words=text.split()
        start=0
        index=0
        while start < len(words):
            end=start + chunk_size
            chunk_words=words[start:end]
            chunk_text=" ".join(chunk_words)
            chunks.append({
                "chunk_id": chunk_counter,
                "text": chunk_text,
                "source": doc["source"],
                "page": doc["page"],
                "chunk_index": index
            })
            chunk_counter +=1
            index +=1
            start=end-overlap
    return chunks
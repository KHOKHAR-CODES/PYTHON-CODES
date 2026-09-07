


from pickle import TRUE


filetype = input("enter a file type:").strip().lower()
if filetype in ("pdf", "docx", "txt"):
    print("supported file type")
else:
    print("unsupported file type")


sentence="python supports rag and document processing"
words=sentence.split()
long_word_count=0
for word in words:
    if len(word) > 5:
        long_word_count += 1
print(f"Number of long words: {long_word_count}")


chunks=["python supports rag and document processing"
        "rag is useful for document processing"
        "rag is useful for retrieval"]
keyword=["rag","retrieval","document"]
word_count=0
for chunk in chunks:
    for key in keyword:
        if key in chunk:
            word_count += 1
print(f"Number of occurrences: {word_count}")
print("total matches:", word_count)

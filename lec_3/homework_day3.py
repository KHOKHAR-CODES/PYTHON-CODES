# Homework Day 3 - Document Processing and Chunking

title = input("Enter document title: ").strip()
source = input("Enter source filename: ").strip()
text = input("Enter a paragraph (at least 15 words): ").strip()

cleaned_text = text.lower()


cleaned_text = cleaned_text.replace(".", "").replace(",", "").replace("!", "").replace("?", "")

words = cleaned_text.split()

char_count = len(cleaned_text)
word_count = len(words)
first_word = words[0]
final_word = words[-1]
python_count = words.count("python")

print("\n--- CLEANED TEXT INFO ---")
print("Cleaned text:", cleaned_text)
print("Character count:", char_count)
print("Word count:", word_count)
print("First word:", first_word)
print("Final word:", final_word)
print('Occurrences of "python":', python_count)


# CHUNKING SECTION
chunk_size = word_count // 3

chunk1_words = words[0:chunk_size]
chunk2_words = words[chunk_size:chunk_size * 2]
chunk3_words = words[chunk_size * 2:word_count]  # remainder goes in last chunk

# Join each chunk of words back into a single string.
chunk1_text = " ".join(chunk1_words)
chunk2_text = " ".join(chunk2_words)
chunk3_text = " ".join(chunk3_words)

# Store the chunk strings together in a list.
chunks = [chunk1_text, chunk2_text, chunk3_text]

# METADATA / DICTIONARY SECTION
document = {
    "title": title,
    "source": source,
    "full_text": cleaned_text,
    "word_count": word_count,
    "char_count": char_count
}

# One dictionary per chunk, each carrying metadata back to the document.
chunk_dict_1 = {
    "chunk_id": 1,
    "text": chunks[0],
    "source": source,
    "title": title
}

chunk_dict_2 = {
    "chunk_id": 2,
    "text": chunks[1],
    "source": source,
    "title": title
}

chunk_dict_3 = {
    "chunk_id": 3,
    "text": chunks[2],
    "source": source,
    "title": title
}

# All chunk dictionaries stored together in one list.
chunk_records = [chunk_dict_1, chunk_dict_2, chunk_dict_3]


print("\n--- DOCUMENT METADATA ---")
print(document)

print("\n--- CHUNK RESULTS ---")
print("Total number of chunks:", len(chunk_records))

print(f"\nChunk {chunk_dict_1['chunk_id']}:")
print("  Title:", chunk_dict_1["title"])
print("  Source:", chunk_dict_1["source"])
print("  Text:", chunk_dict_1["text"])

print(f"\nChunk {chunk_dict_2['chunk_id']}:")
print("  Title:", chunk_dict_2["title"])
print("  Source:", chunk_dict_2["source"])
print("  Text:", chunk_dict_2["text"])

print(f"\nChunk {chunk_dict_3['chunk_id']}:")
print("  Title:", chunk_dict_3["title"])
print("  Source:", chunk_dict_3["source"])
print("  Text:", chunk_dict_3["text"])
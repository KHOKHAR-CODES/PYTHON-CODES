from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")
sentence = [
    "python is usefull",
    "rag is retrivel augment generative",
    "the football match match starts at 8o clock"
]
embedding =model.encode(sentence)
print("number of sentnce:",len(sentence))
print("embeddings shape:",embedding[3:])

#exercise1
text="retrieval"
print(text[:1])
print(text[-1:])
print(text[2:6])
print(text[::-1])
print(len(text))

#excersise2
text="  PYTHON, RAG, AND AI ARE USEFUL!"
clean_text=text.strip().replace(",","").replace("!","").lower()
print(clean_text)
print(len(clean_text))

#excersise3
tools=["python","langchain"]
tools.append("chromaDB")
tools.insert(1,"hugging_face")
tools.extend(["fastapi","streamlit"])
tools.remove("langchain")
print(tools)
print(len(tools))

#excersise4
document = {
 "title": "RAG Notes",
 "pages": 12,
 "processed": False
}
print(document["title"])
document["source"] = "rag_notes.pdf"
document["processed"] = True
document["keywords"] = ["rag", "retrieval", "llm"]
print(len(document))
print(document["keywords"][1])

#excersise5
text = "python helps develoer build pratical ai system with document processing api and retrivel pipeline"
text = text.lower()
words = text.split()
chunk1 = words[0:3]
chunk2 = words[5:10]
chunk3 = words[10:]
chunk1_text = " ".join(chunk1)
chunk2_text = " ".join(chunk2)
chunk3_text = " ".join(chunk3)
print(chunk1_text)
print(chunk2_text)
print(chunk3_text)






document_title="intro to rag" # VARIABLE AND DATA TYPE
page_count=8
relevance_score=0.6
is_processed=False
keywords=["python","java","html"]
doc={"tilt": document_title,
     "pages":page_count}
print(type(document_title))
print(type(keywords))
print(type(doc))

title="python"# review of len()
keywords=["ai","rag","python"]
record={"title":"rag notes","pages":10}
print(len(title))
print(len(keywords))
print(len(record))

city="bahawalur" # creating strings with quotes
country='pakistan'
print(city)
print(country)
print(type(country))


message1="python is'beginner friendly' language" #quotes inside string
message2='THE student said,"rag use doc"'
print(message1)
print(message2)

message="first line\nsecond line" #escape characters in code
path="\\user\\student\\document"
quote="the teacher said \"start coding\""
print(message)
print(path)
print(quote)

document="""Retrieval-Augmented Generation combines retrieval with language-model generation.  #multiline string
 It can answer questions using external documents."""
print(document)

word="usama" # positive indexes
print(word[0])
print(word[3])
print(word[4])

word="usama" # negative indexes
print(word[-1])
print(word[-3])
print(word[-4])

word="python" # index error
print(word[10])

word="usama" # basic index slicing
print(word[0:3])
print(word[3:5])

word="python" # omitting startandend
print(word[:3])
print(word[5:])

word="python" # slicing with steps
print(word[0:3:5])
print(word[1:2:4])

word="python"  #reversing a string
print(word[::-2]) 

text="rag" #g len() with indexing
last_index=len(text)-1
print(len(text))
print(last_index)
print(text[last_index])

first_name="usama" # concatetaon with +
last_name="khokhar"
full_name=first_name+""+last_name
print(full_name)

seprator="_"*20 # repetion with *
print(seprator)

title="rag guide" #f string
pages="25"
message=f"the document {title} contain {pages}"
print(message)

chunk_size=5 #expression inside f string
total_word=13
print(f"words remainig after two chunks:{total_word-chunk_size*2}")

sentence="python is useful for rag" # membership with in or not
print("python"in sentence)
print("java" in sentence)
print("java" not in sentence)

sentence="Python is useful for rag" #case sensitivity
keywords="python"
print(keywords in sentence)
print(keywords.lower() in sentence.lower())

word="python" # strings are immutable
new_word="j"+word[2]
print(word)
print(new_word)

text="PYTHON IS USEFUL FOR RAG" #lowwer()
clean_text=text.lower()
print(clean_text)

text="python" # upper()
print(text.upper())

heading="python is useful for rag" # title()
print(heading.title())

sentence="python is useful for rag"# capitalize()
print(sentence.capitalize())

text="  python is useful for rag   " # strip,lstrip,rstrip
print(text.strip())
print(text.lstrip())
print(text.rstrip())

text="python is difficult" # replace()
updated=text.replace("difficult","practical")
print(updated)

sentence="rag retrives information from document" # slit()
word=sentence.split()
print(word)
print(type(word))
print(len(word))

tools_text="python,langchain,chromdb,fast api" # split()with commas
tools=tools_text.split(",")
print(tools)

words=["RAG", "uses", "external", "documents"] #join()
sentence = " ".join(words)
print(sentence)

tools=["Python", "LangChain", "ChromaDB"] # joining with split
print(", ".join(tools))

sentence="Python is useful for RAG." # find()
print(sentence.find("useful"))
print(sentence.find("Java"))

text="Python is easy. PYTHON is powerful. python is popular."# count()
print(text.count("Python"))
print(text.lower().count("python"))

filename="report.pdf" # startswith()
print(filename.startswith("report"))
print(filename.startswith("notes"))

filename="company_policy.pdf" #endswith()
print(filename.endswith(".pdf"))
print(filename.endswith(".docx"))

text="PYTHON, RAG, AND AI!" 
clean_text = text.strip().lower().replace(",", "").replace("!", "")
print(clean_text)


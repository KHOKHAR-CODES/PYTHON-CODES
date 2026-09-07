# Part 1: Variables and Data Types
print("PART 1: Variables and Data Types")

name = "usama"
age = 19
city = "bahawalpur"
languages = ["Python", "JavaScript", "C++"]
person = {"name": "usama", "age": 19}

print("name:", name)
print("age:", age)
print("city:", city)
print("languages:", languages)
print("person:", person)

print("\n-- type() of each --")
print("type(name):", type(name))
print("type(age):", type(age))
print("type(city):", type(city))
print("type(languages):", type(languages))
print("type(person):", type(person))

print("\n-- len() --")
print("len(name):", len(name))
print("len(languages):", len(languages))
print("len(person):", len(person))

# Part 2: Strings

single_quote_str = 'Hello, this uses single quotes'
double_quote_str = "Hello, this uses double quotes"
print(single_quote_str)
print(double_quote_str)

quote_string = "She said, \"Python is amazing!\""
print(quote_string)

multiline_string = """This is line one.
This is line two.
This is line three."""
print(multiline_string)

escape_demo = "Name:\tusama\nCity:\tbahawalpur\nQuote:\t\"Keep learning\""
print(escape_demo)


# Part 3: String Indexing and Slicing

text = "Artificial Intelligence"

print("First character:", text[0])
print("Last character:", text[-1])
print("Fifth character:", text[4])
print("First 10 characters:", text[:10])
print("Index 11 to end:", text[11:])
print("Every second character:", text[::2])
print("Reversed string:", text[::-1])
print("Last valid index (len-1):", len(text) - 1)


# Part 4: String Operators

first_name = "usama"
last_name = "khokhar"
full_name = first_name + " " + last_name
print("Concatenated name:", full_name)

student_name = "usama"
course = "Python Programming"
semester = "Fall 2026"
info = f"Student {student_name} is enrolled in {course} for {semester}."
print(info)

print("'Python' in course:", "Python" in course)
print("'Java' not in course:", "Java" not in course)


# Part 5: String Methods

text2 = " PYTHON, RAG, and AI are Useful! "

print("Original: '{}'".format(text2))
print("lower():", text2.lower())
print("upper():", text2.upper())
print("title():", text2.title())
print("capitalize():", text2.capitalize())
print("strip():", "'" + text2.strip() + "'")

replaced = text2.replace("Useful", "Powerful")
print("replace('Useful','Powerful'):", replaced)

no_commas = text2.replace(",", "")
print("remove commas:", no_commas)

split_words = text2.split()
print("split():", split_words)

joined_back = " ".join(split_words)
print("join():", joined_back)

print("find('RAG'):", text2.find("RAG"))
print("count('AI'):", text2.count("AI"))
print("startswith('PYTHON'):", text2.strip().startswith("PYTHON"))
print("endswith('!'):", text2.strip().endswith("!"))


# Part 6: Text Cleaning

sentence = "  Python, RAG, and AI, are Useful!  "

cleaned = sentence.strip().lower()
cleaned = cleaned.replace(",", "").replace("!", "")
words = cleaned.split()
clean_text = " ".join(words)

total_characters = len(clean_text)
total_words = len(words)

print("Original sentence:", repr(sentence))
print("Clean text:", clean_text)
print("Total characters:", total_characters)
print("Total words:", total_words)
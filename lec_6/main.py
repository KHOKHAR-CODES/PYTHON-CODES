# main.py
import textutils as tu
# main.py

text = "   Python is useful for RAG.   "
keyword = " PYTHON "

cleaned_text = tu.clean_text(text)
cleaned_keyword = tu.prepare_keyword(keyword)

print(cleaned_text)
print(cleaned_keyword)
print(tu.contains_keyword(cleaned_text, cleaned_keyword))
from langchain_core.output_parsers import StrOutputParser


parser = StrOutputParser()
response = model.invoke("Define Langchain in one sentence")
plain_text = parser.invoke(response)

print("Message object: ")
print(response)

print("\nPlain text: ")
print(plain_text)

print("Parser", parser)
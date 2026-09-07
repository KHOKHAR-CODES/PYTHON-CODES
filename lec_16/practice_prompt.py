from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template(
    "Explain {topic} to a beginner using one example"
)

formatted = prompt.invoke({"topic" : "embeddings"})

print("variable:", prompt.input_variables)
print("Formatted prompt:", formatted)


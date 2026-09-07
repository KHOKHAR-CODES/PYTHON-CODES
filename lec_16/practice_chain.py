from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser



prompt = ChatPromptTemplate.from_messages([
    ("system", "you are a clear AI instructor"),
    ("human", "Explain {topic} in {points} short points")
])

chain = prompt | model | StrOutputParser()

answer = chain.invoke({"topic": "vector databases", "points": 3})

print(answer)



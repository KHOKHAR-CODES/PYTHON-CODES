from langchain_core.prompts import ChatPromptTemplate


study_prompt = ChatPromptTemplate.from_messages([
    ("system", 
    "you are a patient ai instructor"
    "use simple language and do not use advanced mathematics"),
    ("human",
    "Teach the topic below to a {level} student. \n\n"
    "Topic: {topic}\n\n"
    "use this format:\n"

    "1. Simple definition\n"
    "2. Three key points\n"
    "3. one easy example\n"
    "4. two short review questions"

     )


])
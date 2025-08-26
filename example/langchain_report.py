from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_ollama import ChatOllama
from langchain.schema.output_parser import StrOutputParser
import os
from dotenv import load_dotenv  
load_dotenv()

model = ChatOllama(
    model="mistral",
    temperature=0,
    # other params...
)

# prompt
prompt1 = PromptTemplate(
    input_variables=["topic"],
    template="generate a detailed report on {topic}",
)
prompt2 = PromptTemplate(
    input_variables=["text"],
    template="summarize the following text in bullet points: {text}",
)

parser = StrOutputParser()

chain = prompt1 | model | prompt2 | model | parser
result = chain.invoke(
    {"topic": "sun"},
)

print(result)

with open("history.txt", "a+") as f:
    f.write(f"{result}\n")
# prompt
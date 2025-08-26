from langchain_ollama import ChatOllama
import os
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain.schema.output_parser import StrOutputParser
from dotenv import load_dotenv 

load_dotenv()

model = ChatOllama(
    model="mistral",
    temperature=0,
    # other params...
)

# Set environment variables for LangSmith tracing and API key   
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

chat_history = []
ai_history = []

while True:
    user_input = input("You: ")
    chat_history.append(HumanMessage(content=user_input)) 
    if user_input == "exit":
        break
    result = model.invoke(user_input)
    print("AI: ", result.content)
    ai_history.append(AIMessage(content=result.content))

print("Chat History: ", chat_history)
print("AI History: ", ai_history)


with open("history.txt", "a+") as f:
    # chat_history.append("\n"f.write("Chat History: \n"))
    # ai_history.append("\n"f.write("AI History: \n"))
    for chat in chat_history:
        f.write(f"Chat History : {chat}\n")
    for ai in ai_history:
        f.write(f"AI History : {ai}\n")

    
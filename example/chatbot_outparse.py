from langchain_ollama import ChatOllama
import os
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain.schema import AIMessage
from langchain.schema.output_parser import StrOutputParser
from dotenv import load_dotenv

load_dotenv()   

llm = ChatOllama(
    model="mistral",
    temperature=0,
    # other params...
)

StrOutputParser()
chat_history = []
ai_history = []

while True:
    user_input = input("You: ")
    chat_history.append(StrOutputParser().parse(user_input))
    if user_input == "exit":
        break
    result = llm.invoke(user_input)
    print("AI: ", result.content)
    ai_history.append(StrOutputParser().parse(result.content))

print("Chat History: ", chat_history)
print("AI History: ", ai_history)

with open("history.txt", "a+") as f:
    for chat in chat_history:
        f.write(f"Chat History : {chat}\n")
    for ai in ai_history:
        f.write(f"AI History : {ai}\n")
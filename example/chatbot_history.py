from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_ollama import ChatOllama
import os
from dotenv import load_dotenv  
load_dotenv()

# chat template
chat_template = ChatPromptTemplate([
    ("system", "You are a helpful assistant"),
    MessagesPlaceholder(variable_name="chat_history"),  # chat history
    ("human", "{query}")  # user input

]
)
chat_history = []
ai_history = []
# chat history
with open("history.txt", "r") as f:
    chat_history.extend(f.readlines())

print("Chat History: ", chat_history)
print("AI History: ", ai_history)


# chat prompt
prompt = chat_template.invoke({"chat_history": chat_history, "query": "capital of india?"})

print("Prompt: ", prompt)

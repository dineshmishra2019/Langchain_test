import langchain_ollama
from langchain_ollama import ChatOllama
import os
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain.schema import AIMessage
from langchain.schema.output_parser import StrOutputParser
from dotenv import load_dotenv

load_dotenv()   
# Set environment variables for LangSmith tracing and API key   
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

# Initialize the Ollama LLM
llm = ChatOllama(
    model="mistral",
    temperature=0,
    # other params...
)

st.header("LangChain + Ollama + Streamlit")
user_input = st.text_input("Enter your question here", key="input")

if st.button("Submit"):
    response = llm.invoke(user_input)
    st.write(response.content)

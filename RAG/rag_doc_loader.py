from langchain_ollama import ChatOllama
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
import os
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain.schema import AIMessage
from langchain.schema.output_parser import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

loader = TextLoader("history.txt", encoding="utf8")
documents = loader.load()

llm = ChatOllama(
    model="mistral",
    temperature=0.7,
    # other params...
)

prompt = PromptTemplate(
    input_variable = 'input',
    template = "write a detailed summary of the following text: {input}"

)

parser = StrOutputParser()

# print(documents)

print(documents[0].page_content)
print(documents[0].metadata)
print(len(documents))
print(type(documents))
print(type(documents[0]))

chain = prompt | llm | parser

result = chain.invoke(
    {
        "input": documents[0].page_content
    }
)

print(result)
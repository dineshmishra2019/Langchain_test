from langchain_ollama import ChatOllama
from langchain_community.document_loaders import CSVLoader, WebBaseLoader, TextLoader, PyPDFLoader, PyPDFDirectoryLoader, Docx2txtLoader, UnstructuredWordDocumentLoader, PyMuPDFLoader
from langchain.text_splitter import CharacterTextSplitter
import os
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain.schema import AIMessage
from langchain.schema.output_parser import StrOutputParser
from dotenv import load_dotenv
load_dotenv()

loader = CSVLoader(file_path='spotify_history.csv')
docs = loader.load()

print(docs[2])
print(len(docs))


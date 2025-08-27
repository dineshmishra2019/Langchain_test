from langchain_ollama import ChatOllama
from langchain_community.document_loaders import TextLoader, PyPDFLoader, PyPDFDirectoryLoader, Docx2txtLoader, UnstructuredWordDocumentLoader, PyMuPDFLoader
from langchain.text_splitter import CharacterTextSplitter
import os
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain.schema import AIMessage
from langchain.schema.output_parser import StrOutputParser
from dotenv import load_dotenv
load_dotenv()

llm = ChatOllama(
    model="mistral",
    temperature=0.7,
    # other params...
)
loader = PyMuPDFLoader("dinesh.pdf")
documents = loader.lazy_load()

for document in documents:
    print(document.page_content)
    print(document.metadata)
    
    


# print(documents[4].page_content)
# print(documents[0].metadata)
# print(len(documents))
# print(type(documents))
# print(type(documents[0]))
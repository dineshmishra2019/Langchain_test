from langchain_ollama import ChatOllama
from langchain_community.document_loaders import WebBaseLoader, TextLoader, PyPDFLoader, PyPDFDirectoryLoader, Docx2txtLoader, UnstructuredWordDocumentLoader, PyMuPDFLoader
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

prompt = PromptTemplate(
    input_variable = 'input',
    template = "write a summary of the following text: {input}"

)

parser = StrOutputParser()

url = 'https://www.w3schools.com/django/index.php'

loader = WebBaseLoader(url)
web1 = loader.load()

chain = prompt | llm | parser

print(chain.invoke(
    {
        "input":web1[0].page_content
    }
))



# for web in web1:
#     print(web.page_content)
#     print(web.metadata)
# print(len(web1))
# print(type(web1))
# print(type(web1[0]))



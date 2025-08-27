from langchain_community.retrievers import WikipediaRetriever
from langchain_community.document_loaders import WebBaseLoader, TextLoader, PyPDFLoader, PyPDFDirectoryLoader, Docx2txtLoader, UnstructuredWordDocumentLoader, PyMuPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_ollama import ChatOllama
import os
from dotenv import load_dotenv

load_dotenv()

llm = ChatOllama(
    model="mistral",
    temperature=0.7,
    # other params...
)


retriver = WikipediaRetriever(top_k=2, lang='en')

query = "Summurize Top 5 business tycoon in india"

docs = retriver.invoke(query)

for i, doc in enumerate(docs):
    print(f" Result No. {i+1}:>>>>>>>>>>>>>>>> ")
    print(f" Contents are :   ",doc.page_content)

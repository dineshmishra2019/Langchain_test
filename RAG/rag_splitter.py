from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader, TextLoader, PyPDFLoader, PyPDFDirectoryLoader, Docx2txtLoader, UnstructuredWordDocumentLoader, PyMuPDFLoader
#from langchain.core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain.schema import AIMessage
from langchain.schema.output_parser import StrOutputParser
from langchain_ollama import ChatOllama
import os
from dotenv import load_dotenv

load_dotenv()

llm = ChatOllama(
    model="mistral",
    temperature=0.7,
    # other params...
)

# loader = PyMuPDFLoader("dinesh.pdf")
# documents = loader.lazy_load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 486,
    chunk_overlap=0

)




# splitter = CharacterTextSplitter(
#     chunk_size = 300,
#     separator = " ",
#     chunk_overlap=5
# )
# result = splitter.split_documents(documents)
# print(result[0].page_content)
# print(len(result))



# for document in documents:
#     print(document.page_content)
    


text = """
    Learning the parameters of a prediction function and testing it on the same data is a methodological mistake: a model that would just repeat the labels of the samples that it has just seen would have a perfect score but would fail to predict anything useful on yet-unseen data. This situation is called overfitting. To avoid it, it is common practice when performing a (supervised) machine learning experiment to hold out part of the available data as a test set X_test, y_test. Note that the word “experiment” is not intended to denote academic use only, because even in commercial settings machine learning usually starts out experimentally. Here is a flowchart of typical cross validation workflow in model training. The best parameters can be determined by grid search techniques.
"""


result = splitter.split_text(text)
print(result)
print(len(result))


# splitter = CharacterTextSplitter(
#     chunk_size = 50,
#     separator = " ",
#     chunk_overlap=5
# )

# result =splitter.split_text(text)
# print(result)
# print(len(result))



from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.document_loaders import WebBaseLoader, TextLoader, PyPDFLoader, PyPDFDirectoryLoader, Docx2txtLoader, UnstructuredWordDocumentLoader, PyMuPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_ollama import ChatOllama
#from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

import os
from dotenv import load_dotenv


load_dotenv()
# os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")



llm = ChatOllama(
    model="mistral",
    temperature=0.7,
    # other params...
)
embeddings = OllamaEmbeddings(model="llama3")

text_splitter = SemanticChunker(
    embeddings, 
    breakpoint_threshold_type = 'standard_deviation', 
    breakpoint_threshold_amount=1
)
    
sample = """
     The Sun is a star located in the Milky Way galaxy and at the center of our solar system.
- It has a diameter approximately 109 times larger than Earth's, with a radius of about 696,000 kilometers (432,000 miles).
- The Sun's surface temperature is around 5,500 degrees Celsius (9,932 degrees Fahrenheit), while its core is estimated to be about 15,000,000 degrees Celsius (27,000,000 degrees Fahrenheit).
- Learning the parameters of a prediction function and testing it on the same data is a methodological mistake: a model that would just repeat the labels of the samples that it has just seen would have a perfect score but would fail to predict anything useful on yet-unseen data. This situation is called overfitting. To avoid it, it is common practice when performing a (supervised) machine learning experiment to hold out part of the available data as a test set X_test, y_test. Note that the word “experiment” is not intended to denote academic use only, because even in commercial settings machine learning usually starts out experimentally. Here is a flowchart of typical cross validation workflow in model training. The best parameters can be determined by grid search techniques.- 
- The Sun consists of several layers: the core, radiative zone, convective zone, photosphere, chromosphere, and corona.
- The Sun's energy is produced through nuclear fusion in its core.

"""

docs = text_splitter.create_documents([sample])
# print(docs)
print(len(docs))
print(type(docs))


vectorstore = FAISS.from_documents(docs, embeddings)

query1 = "What does this document say about sun?"
query2 = "What does this document say about Learning?"
results = vectorstore.similarity_search(query2, k=4)
print(results)

for r in results:
    print("\n--- Result ---")
    print(r.page_content)


from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS, Chroma
#from langchain_core.vectorstores import VectorStoreIndexWrapper, VectorStore
#from langchain_community.document_loaders import DocumnetLoader, WebBaseLoader, TextLoader, PyPDFLoader, PyPDFDirectoryLoader, Docx2txtLoader, UnstructuredWordDocumentLoader
from langchain.text_splitter import CharacterTextSplitter   
#from langchain_openai import OpenAIEmbeddings
#from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
#from langchain.document_loaders import PyPDFLoader, PyMuPDFLoader
from langchain.schema import Document


embeddings = OllamaEmbeddings(
    model="llama3",
)

doc1 = Document(page_content=" reena is the captain of RCB", metadata={"team":"RCB Team"})
doc2 = Document(page_content="Dhoni is the captain of Chennai", metadata={"team":"Channai Team"})
doc3 = Document(page_content="chris gale is the captain of rajasthan", metadata={"team":"Rajasthan Team"})
doc4 = Document(page_content="ravinder is the captain of paris", metadata={"team":"Paris Team"})
doc5 = Document(page_content="ganga is the captain of usa", metadata={"team":"USA Team"})
doc6 = Document(page_content="praveen is the captain of India", metadata={"team":"INDIA Team"})
doc7 = Document(page_content="shankar is the captain of xyz", metadata={"team":"CHINA Team"})

docs = [doc1, doc2, doc3, doc4, doc5, doc6, doc7]
vector_store = Chroma(persist_directory="./chroma-db", embedding_function=OllamaEmbeddings(model="llama3"), collection_name="sample")
# vector_store = FAISS.from_documents(docs, embeddings)
#vector_store.persist()
print(vector_store.add_documents(docs))

# print(vector_store.get(include=["embeddings", "documents", "metadatas"]))

# update_doc8=Document(page_content="sharma is new player", metadata={"team":"RCB Team"})

# print(vector_store.update_document(document_id='0bd1426e-0da3-4403-98ce-76ce74fa9124', document=update_doc8))
print(vector_store.get(include=["embeddings", "documents", "metadatas"]))
print(vector_store.similarity_search(
    query='USA',
    k=1
))
print(vector_store.similarity_search_with_score(
    query='USA',
    k=1
))



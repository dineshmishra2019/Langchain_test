from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_ollama import ChatOllama
import os
from dotenv import load_dotenv

load_dotenv()

model = ChatOllama(
    model="mistral",
    temperature=0,
    # other params...
)
# Set environment variables for LangSmith tracing and API key   
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

messages = [
    SystemMessage(content="You are a helpful assistant that translates English to French. Translate the user sentence."),
    HumanMessage(content="I love programming."),
    AIMessage(content="J'adore programmer.")
]
Result = model.invoke(messages)
print("Response: ", Result.content)

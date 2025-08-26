from langchain_ollama import ChatOllama
import os
#from langchain_core.chains import LLMChain
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain.schema import AIMessage
from langchain.schema.output_parser import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

model = ChatOllama(
    model="mistral",
    temperature=0,
    # other params...
)

# prompt = ChatPromptTemplate.from_messages(
#     [
#         (
#             "system",
#             "You are a helpful assistant that translates {input_language} to {output_language}.",
#         ),
#         ("human", "{input}"),
#     ]
# )
prompt = PromptTemplate(
    input_variables=["input_language", "output_language", "input"],
    template="You are a helpful assistant that translates {input_language} to {output_language}. Translate the user sentence. {input}",
)
output_parser=StrOutputParser()

chain = prompt | model | output_parser
result = chain.invoke(
    {
        "input_language": "English",
        "output_language": "French",
        "input": "I love programming. but unable to do it",
    }
)

print(result)
with open("history.txt", "a+") as f:
    f.write(f"{result}\n")
    f.write(chain.get_graph().print_ascii())
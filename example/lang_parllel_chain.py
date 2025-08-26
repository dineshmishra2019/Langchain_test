from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable  import RunnableParallel

import os
from dotenv import load_dotenv  
load_dotenv()

model1 = ChatOllama(
    model="mistral",
    temperature=0,
    # other params...
)
model2 = ChatOllama(
    model="llama2",
    temperature=0,
    # other params...
)

prompt1 = PromptTemplate(
    input_variables=["topic"],
    template="generate a detailed report on {topic}",
)
prompt2 = PromptTemplate(
    input_variables=["topic"],
    template="generate 5 question and answers on the : {topic}",
)

parser = StrOutputParser()



parellel_chain = RunnableParallel(
    {
        'Run1' : prompt1 | model1 | parser,
        'Run2' : prompt2 | model2 | parser }
)

response = parellel_chain.invoke({"topic": "sun"})
text = response['Run1'] + "\n" + response['Run2']


prompt3 = PromptTemplate(
    input_variables=["text"],
    template="summarize the following text in bullet points: {text}"
)

chain = prompt3 | model1 | parser

result = chain.invoke(
    {"text": text}
)

print(result)

# # # 
# # def record(result):
# #     with open("history.txt", "a+") as f:
# #         f.write(f"{result}\n")

with open("history.txt", "a+") as f:
    f.write(f"{result}\n")

from langchain_ollama import ChatOllama
import os
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain.schema import AIMessage
from langchain.schema.output_parser import StrOutputParser
from dotenv import load_dotenv

load_dotenv()


# Set environment variables for LangSmith tracing and API key   

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

# Initialize the Ollama LLM
llm = ChatOllama(
    model="mistral",
    temperature=0,
    # other params...
)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant that translates {input_language} to {output_language}.",
        ),
        ("human", "{input}"),
    ]
)
st.title("Translater app")
input_txt=st.text_input("Enter the input to translate to french")
st.write("You entered: ", input_txt)

output_parser=StrOutputParser()

chain = prompt | llm | output_parser | prompt
if input_txt:
    response=chain.invoke({"input":input_txt,"input_language":"English","output_language":"French"})
    st.write(response)

# result = chain.invoke(
#     {
#         "input_language": "English",
#         "output_language": "German",
#         "input": "I love programming.",
#     }
# )

# print(result.content)

# messages = [
#     (
#         "system",
#         "You are a helpful assistant that translates English to French. Translate the user sentence.",
#     ),
#     ("human", "I love programming."),
# ]
# ai_msg = llm.invoke(messages)

#AIMessage(content='The translation of "I love programming" in French is:\n\n"J\'adore le programmation."', additional_kwargs={}, response_metadata={'model': 'llama3.1', 'created_at': '2025-06-25T18:43:00.483666Z', 'done': True, 'done_reason': 'stop', 'total_duration': 619971208, 'load_duration': 27793125, 'prompt_eval_count': 35, 'prompt_eval_duration': 36354583, 'eval_count': 22, 'eval_duration': 555182667, 'model_name': 'llama3.1'}, id='run--348bb5ef-9dd9-4271-bc7e-a9ddb54c28c1-0', usage_metadata={'input_tokens': 35, 'output_tokens': 22, 'total_tokens': 57})



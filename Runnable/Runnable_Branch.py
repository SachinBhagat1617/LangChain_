# Runnable Branch allows conditional execution of runnables based on input
# This is useful for creating dynamic pipelines that adapt based on input parameters

from langchain_core.runnables import RunnableBranch,RunnableParallel,RunnablePassthrough
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
load_dotenv()
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation",
)

model = ChatHuggingFace(llm=llm)

prompt1=PromptTemplate(
    template="Write a report of the following Topic: {topic}",    
    input_variables=['topic']
)
parser= StrOutputParser()

prompt2= PromptTemplate(
    template="Summarise the text {text}",
    input_variables=['text']
)

chain= prompt1 | model | parser

final_chain= RunnableBranch(
    (lambda x:len(x.split()) > 200, prompt2 | model | parser),
    RunnablePassthrough()
)
full_chain= chain | final_chain
response = full_chain.invoke({'topic':'Climate Change and its impact on global economy'})
print(response)
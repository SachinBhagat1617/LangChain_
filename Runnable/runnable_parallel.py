from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation",
)
model = ChatHuggingFace(llm=llm)
prompt1=PromptTemplate(
    template="Write a X post on the following Topic: {topic}",
    input_variables=['topic']
)
parser= StrOutputParser()
prompt2=PromptTemplate(
    template="Generate a linkedIn post content : {topic}",
    input_variables=['topic']
)

chain= RunnableParallel(
    {
        "X":prompt1 | model |parser , 
        "LinkedIn":prompt2 | model |parser
    }
)
response = chain.invoke({'topic':'Langchain with Mistral LLM'})

print(response)
# Runnable Passthrough gives same input as output
# UseCase in chaining where intermediate step needs original input
# for example in logical branching based on input parameters
# in prev example of explaining joke, we may want to pass topic along with joke explanation
# so for that we can use RunnablePassthrough to pass topic alongside joke explanation step 
# so that both joke explanation and topic are available for next steps

from langchain_core.runnables import RunnablePassthrough,RunnableParallel
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
    template="Write a joke of the following Topic: {topic}",
    input_variables=['topic']
)
parser= StrOutputParser()
prompt2= PromptTemplate(
    template="Explain the following joke in simple terms: {joke}",
    input_variables=['joke']
)

parallel_chain= RunnableParallel(
    {
        "topic": RunnablePassthrough(),
        "joke": prompt2 | model | parser
        
    }
)
chain = prompt1 | model | parser | parallel_chain
response = chain.invoke({'topic':'About AI'})

print(response)
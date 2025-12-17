# Runnable Lambda allows defining custom functions as runnables
# This is useful for simple transformations or operations that don't require full prompt templates or models
# and is used for creating small reusable components in a pipeline

from langchain_core.runnables import RunnableLambda,RunnablePassthrough, RunnableParallel
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

def count_words(text: str) -> int:
    """Count the number of words in the input text."""
    return len(text.split())

chain= prompt1 | model | parser
parallel_chain= RunnableParallel(
    {
       "joke": RunnablePassthrough(),
        "word_count": RunnableLambda(count_words)
    }
)
final_chain= chain | parallel_chain
response = final_chain.invoke({'topic':'Programming'})
print(response)
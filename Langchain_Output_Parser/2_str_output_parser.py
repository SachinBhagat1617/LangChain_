from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation",
)

model =ChatHuggingFace(llm=llm)

# Output parser is used to form structured output from the model's response for opensource models which generally give unstructured text output

#1st prompt
template1=PromptTemplate(
    template="Write detailed report on the following topic: {topic}",
    input_variables=["topic"]
)
#2nd prompt with 
template2=PromptTemplate(
    template="Summarize the following text in bullet points:\n\n{text}",
    input_variables=["text"]
)
# output parser instance
parser=StrOutputParser()

chain=template1 | model | parser | template2 | model | parser

result=chain.invoke({"topic":"Artificial Intelligence"})
print("\nSummary in Bullet Points:\n", result)
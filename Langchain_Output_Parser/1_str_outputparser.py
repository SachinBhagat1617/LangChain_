from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

# without output parser these open source models may generate unstructured text
# we can use output parsers to structure the output but here we are just generating text without any output parser

llm=HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation",
)

model =ChatHuggingFace(llm=llm)

template1=PromptTemplate(
    template="Write detailed report on the following topic: {topic}",
    input_variables=["topic"]
)
prompt1=template1.invoke({"topic":"Artificial Intelligence"})
text = model.invoke(prompt1).content

# passing the generated text to model again with different prompt

template2=PromptTemplate(
    template="Summarize the following text in bullet points:\n\n{text}",
    input_variables=["text"]
)
prompt2=template2.invoke({"text":text})
summary = model.invoke(prompt2).content
print("\nSummary in Bullet Points:\n", summary) 



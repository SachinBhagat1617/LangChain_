from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv
load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation",
)

model =ChatHuggingFace(llm=llm)

parser=JsonOutputParser()

# the biggest drawback in JsonOutputParser is that you can not define your own schema like pydantic models
# it just ensures that the output is in json format

template= PromptTemplate(
    template="Generate a JSON object with 'title' and 'description' for a blog post about {topic}. \n {format_instructions}",
    input_variables=["topic"],
    partial_variables={'format_instructions': parser.get_format_instructions()}
)
print(template)
""" input_variables=['topic'] input_types={} partial_variables={'format_instructions': 'Return a JSON object.'} template="Gene
rate a JSON object with 'title' and 'description' for a blog post about {topic}. \n {format_instructions}" """
# format instruction is replaced by parser's get_format_instructions method output Return a JSON object.

chain= template | model | parser
result=chain.invoke({"topic":"Artificial Intelligence"})
print("\nJSON Output:\n", result)
# In pydantic output parser we define a pydantic model to structure the output
# we can enforece the schema, constraints, data types etc.

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation",
)
model = ChatHuggingFace(llm=llm)

class Person(BaseModel):
    name: str = Field(description="Full name of the person")
    age: int = Field(ge=0, le=120, description="Age of the person in years")
    city: str = Field(description="City where the person lives")

parser = PydanticOutputParser(pydantic_object=Person)
template = PromptTemplate(
    template="Generate a person's details including 'name', 'age', and 'city' of a fictional {place} person in the following format:\n{format_instructions}",
    input_variables=['place'],
    partial_variables={'format_instructions': parser.get_format_instructions()}
)

chain= template | model | parser
result=chain.invoke({"place":"Tokyo"})  
#print("\nPydantic Object:\n", result)
print("\n As JSON:\n", result.model_dump_json())
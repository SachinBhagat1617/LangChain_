from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.runnables import RunnableBranch, RunnableLambda
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation",
)
model = ChatHuggingFace(llm=llm)

class Feedback(BaseModel):
    sentiment : Literal["Pos","Neg"] = Field(description="The sentiment of the feedback: positive, negative, or neutral.")
    
parser= StrOutputParser()
parser2= PydanticOutputParser(pydantic_object=Feedback)

prompt1 = PromptTemplate(
    template="Classify the sentiment of the following feedback as Pos, Neg \nFeedback: {feedback} \n {format_instructions}",
    input_variables=['feedback'],
    partial_variables={'format_instructions': parser2.get_format_instructions()}
)

classify_chain = prompt1 | model | parser2

prompt2= PromptTemplate(
    template="If the feedback sentiment is positive, generate a thank you response. Feedback sentiment: {feedback}",
    input_variables=['feedback']
)
prompt3=PromptTemplate(
    template="If the feedback sentiment is negative, generate an apology response and ask how to improve. Feedback sentiment: {feedback}",
    input_variables=['feedback']
)

"""
branch_chain=RunnableBranch(
    (function , chian1)  # if condition is met
    (function , chain2) # else if
    default  # else
)
        
"""

branch_chain=RunnableBranch(
    (lambda x: x.sentiment == "Pos", prompt2 | model | parser),
    (lambda x: x.sentiment == "Neg", prompt3 | model | parser),
    RunnableLambda(lambda x: "No response needed for neutral feedback.")
)
chain= classify_chain | branch_chain
response = chain.invoke({"feedback":"The product quality has improved significantly, and I'm very satisfied with my purchase!"})
print(response)
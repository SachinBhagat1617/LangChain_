from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt1 = PromptTemplate(
    template='Generate a detailed report on the following topic: {topic}',
    input_variables=['topic']
) 

prompt2= PromptTemplate(
    template='Summarize the following report in 5 bullet points: {report}',
    input_variables=['report']
)
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

parser= StrOutputParser()

chain =prompt1 | model | parser | prompt2 | model | parser

response = chain.invoke({'topic': 'The impact of climate change on global agriculture.'})
print(response)

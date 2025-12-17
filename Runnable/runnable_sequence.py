from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence

load_dotenv()

#model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation",
)
model = ChatHuggingFace(llm=llm)

prompt1= PromptTemplate(
    template="Write a joke of the following Topic: {topic}",
    input_variables=['topic']
)
parser= StrOutputParser()
prompt2= PromptTemplate(
    template="Explain the following joke in simple terms: {joke}",
    input_variables=['joke']
)
chain= RunnableSequence(prompt1,model,parser,prompt2,model,parser)
response = chain.invoke({'topic':'Gemini and ChatGPT'})
print(response)
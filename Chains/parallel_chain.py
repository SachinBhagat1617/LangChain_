from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

prompt1 = PromptTemplate(
    template='Generate a detailed notes on the following topic: {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Prepare a quiz with 5 questions based on the following notes: {notes}',
    input_variables=['notes']
)

parser= StrOutputParser()

prompt3= PromptTemplate(
    template="Merge the following notes and quiz into a single study guide.\nNotes: {notes}\nQuiz: {quiz}",
    input_variables=['notes', 'quiz']
)

ParallelChain = RunnableParallel(
    {
        'notes' : prompt1 | model | parser,
        'quiz' : prompt2 | model | parser
    }
)
chain= ParallelChain | prompt3 | model | parser
response = chain.invoke('The fundamentals of quantum mechanics.')
print(response)
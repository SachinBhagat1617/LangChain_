# pip install langchain-community
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation",
)
model = ChatHuggingFace(llm=llm)

prompt=PromptTemplate(
    template="Summarize the following document content: {content}",    
    input_variables=['content']
)
parser= StrOutputParser()
loader=TextLoader('C:\\Users\\sacbhaga\\Desktop\\Langchain\\Langchain_Document_Loaders\\cricket.txt', encoding='utf-8')
documents=loader.load()
document_content= documents[0].page_content
chain= prompt | model | parser
response = chain.invoke({'content':document_content})
print(response)
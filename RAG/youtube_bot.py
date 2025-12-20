from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnableParallel,RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.documents import Document
load_dotenv()

#Load YouTube Transcript
def get_youtube_transcript(video_id):
    try:
        ytt = YouTubeTranscriptApi()
        transcript_list = YouTubeTranscriptApi.fetch(self=ytt,video_id=video_id,languages=['en'])
        transcript = " ".join([entry.text for entry in transcript_list])
        #print("Transcript fetched successfully,",transcript)
        return transcript
    except Exception as e:
        print(f"Error fetching transcript: {e}")
        return None

# Split Transcript into Chunks (text splitting)
def split_transcript(transcript):
    text_splitter=RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,        
    )
    chunks = text_splitter.split_text(transcript)
    documents = [Document(page_content=chunk) for chunk in chunks]
    return documents

# Create Embeddings and Vector Store
def create_vector_store(chunks):
    embed_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store=Chroma.from_documents(chunks,embed_model) # Create Chroma vector store by converting to embeddings
    return vector_store

def retrieve_similar_chunks(vector_store, query, k=2):
    retiever=vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k":k, "lambda_mult":0.5}
    )
    return retiever.invoke(query) # return list of similar documents
    

def formated_response(docs):
    context_text="\n\n".join([doc.page_content for doc in docs])
    return context_text
    
def generate_prompt(context, query):
    prompt=PromptTemplate(
        template="Using the following context, answer the question:\n\nContext: {context}\n\nQuestion: {query}\n\nAnswer:   , if the answer is not found in the context, respond with 'Information not available in the video.'",
        input_variables=['context','query']
    )
    return prompt.format(context=context, query=query)

if __name__ == "__main__":
    video_id="hDKCxebp88A"  # Example YouTube video ID
    transcript=get_youtube_transcript(video_id)
    chunks=split_transcript(transcript)
    vector_store=create_vector_store(chunks)
    query="What's the topic discussed in the video?"
    parallel_chain=RunnableParallel({
        "context":RunnableLambda(lambda x: retrieve_similar_chunks(vector_store, x["query"], k=3))
                    | RunnableLambda(formated_response),
        "query": RunnablePassthrough() 
    })
    result=parallel_chain.invoke({"query":query})
    # Print the context and query
    # print("Context:\n", result["context"])
    # print("\nQuery:\n", result["query"])
    
    parser= StrOutputParser()
    llm= HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.2",
        task="text-generation",
    )
    model= ChatHuggingFace(llm=llm)
    final_chain= parallel_chain | RunnableLambda(lambda x: generate_prompt(x["context"], x["query"])) | model | parser
    final_response=final_chain.invoke({"query":query})
    print("Final Response:\n", final_response)

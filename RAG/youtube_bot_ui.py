import streamlit as st
import re
from youtube_transcript_api import YouTubeTranscriptApi
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

load_dotenv()


def extract_video_id(url):
    match = re.search(r"v=([^&]+)", url)
    return match.group(1) if match else None


def get_youtube_transcript(video_id):
    ytt = YouTubeTranscriptApi()
    transcript_list = ytt.fetch(video_id=video_id, languages=["en"])
    return " ".join(entry.text for entry in transcript_list)


def split_transcript(transcript):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_text(transcript)
    return [Document(page_content=chunk) for chunk in chunks]


def create_vector_store(chunks):
    embed_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return Chroma.from_documents(chunks, embed_model)


def retrieve_similar_chunks(vector_store, query, k=3):
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "lambda_mult": 0.5}
    )
    return retriever.invoke(query)


def formatted_response(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def generate_prompt(context, query):
    prompt = PromptTemplate(
        template="""
Using the following context, answer the question.

Context:
{context}

Question:
{query}

If the answer is not found in the context, say:
"Information not available in the video."
""",
        input_variables=["context", "query"]
    )
    return prompt.format(context=context, query=query)


# STREAMLIT UI


st.set_page_config(page_title="YouTube RAG Bot")
st.title("🎥 YouTube Video Q&A (LangChain Chaining)")

youtube_url = st.text_input("Enter YouTube URL")
query = st.text_input("Ask a question")

if st.button("Get Answer"):
    if not youtube_url or not query:
        st.warning("Please enter both URL and question.")
    else:
        with st.spinner("Building RAG pipeline..."):
            try:
                video_id = extract_video_id(youtube_url)

                transcript = get_youtube_transcript(video_id)
                chunks = split_transcript(transcript)
                vector_store = create_vector_store(chunks)

                # SAME CHAINING CONCEPT AS YOUR SCRIPT
                parallel_chain = RunnableParallel({
                    "context": RunnableLambda(
                        lambda x: retrieve_similar_chunks(
                            vector_store, x["query"], k=3
                        )
                    ) | RunnableLambda(formatted_response),
                    "query": RunnablePassthrough()
                })

                llm = HuggingFaceEndpoint(
                    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
                    task="text-generation"
                )
                model = ChatHuggingFace(llm=llm)

                final_chain = (
                    parallel_chain
                    | RunnableLambda(
                        lambda x: generate_prompt(x["context"], x["query"])
                    )
                    | model
                    | StrOutputParser()
                )

                answer = final_chain.invoke({"query": query})

                st.success("Answer")
                st.write(answer)

            except Exception as e:
                st.error(f"Error: {e}")

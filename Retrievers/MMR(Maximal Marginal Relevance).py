from langchain_core.documents import Document
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

docs = [
    Document(page_content="LangChain makes it easy to work with LLMs."),
    Document(page_content="LangChain is used to build LLM based applications."),
    Document(page_content="Chroma is used to store and search document embeddings."),
    Document(page_content="Embeddings are vector representations of text."),
    Document(page_content="MMR helps you get diverse results when doing similarity search."),
    Document(page_content="LangChain supports Chroma, FAISS, Pinecone, and more."),
]

# Create a vector store from documents
vector_store=Chroma.from_documents(
    documents=docs,
    embedding=embed_model
)

# Convert vector store into a MMR retriever
retriever=vector_store.as_retriever(
    search_type="mmr",  # this enables MMR search
    search_kwargs={"k":3, "lambda_mult":0.5}  # k is number of docs to return, lambda_mult controls diversity context it varies between 0 and 1
)

query="What is LangChain ?"
docs=retriever.invoke(query)
for i,doc in enumerate(docs):
    print(f"Document {i+1}: {doc.page_content}")
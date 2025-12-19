from langchain_community.vectorstores import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

documents = [
    Document(page_content="LangChain helps developers build LLM applications easily."),
    Document(page_content="Chroma is a vector database optimized for LLM-based search."),
    Document(page_content="Embeddings convert text into high-dimensional vectors."),
    Document(page_content="OpenAI provides powerful embedding models."),
]

#Create a vector store from documents
vectorStore=Chroma.from_documents(
    documents=documents,
    embedding=embed_model,
    collection_name="langchain_docs"
)

query="What is LangChain?"

# Convert vector store into a retriever
retriever=vectorStore.as_retriever(search_type="similarity", search_kwargs={"k":2})

docs=retriever.invoke(query)
for i,doc in enumerate(docs):
    print(f"Document {i+1}: {doc.page_content}")
    

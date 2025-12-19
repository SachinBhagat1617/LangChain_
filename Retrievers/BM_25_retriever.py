from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
documents = [
    Document(page_content="LangChain helps developers build LLM applications easily."),
    Document(page_content="BM25 is a popular algorithm for information retrieval."),
    Document(page_content="Embeddings convert text into high-dimensional vectors."),
    Document(page_content="OpenAI provides powerful embedding models."),
    Document(page_content="Chroma is a vector database optimized for LLM-based search."),
]
# Create a BM25 retriever from documents
retriever = BM25Retriever.from_documents(documents,k=1)
query = "What is BM25 ?"
docs = retriever.invoke(query)
for i, doc in enumerate(docs):
    print(f"Document {i+1}: {doc.page_content}")

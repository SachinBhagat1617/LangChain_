from langchain_community.retrievers import WikipediaRetriever

retriever=WikipediaRetriever(top_k_results=2, lang="en")

query="the geopolitical tension between India and Pakistan"
docs=retriever.invoke(query)
for doc in docs:
    print(doc.page_content)
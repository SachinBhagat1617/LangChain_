from langchain_community.document_loaders import PyPDFLoader

loader=PyPDFLoader('C:\\Users\\sacbhaga\\Desktop\\Langchain\\Langchain_Document_Loaders\\dl-curriculum.pdf')
documents=loader.load()
print(len(documents))
print(documents[0].page_content)
print(documents[0].metadata)
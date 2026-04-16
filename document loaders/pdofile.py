from langchain_community.document_loaders import PyPDFLoader

doc = PyPDFLoader("document loaders/Assignment 1.pdf")
data = doc.load()

print(len(data))
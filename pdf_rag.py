from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
load_dotenv()

llm = ChatGoogleGenerativeAI(
    model = 'gemini-3-flash-preview',
    temperature = 0.3
)
system = 'You are data retriever. Find data exactly as asked or describe by the user.'
prompt = ChatPromptTemplate.from_messages([
    ('system', system + "\nContext: {data}"),
    ('user', '{query}')
])

doc = PyPDFLoader('document loaders\Assignment 1.pdf')
data = doc.load()


chain = prompt | llm | StrOutputParser()

user_input = input("You: ")

result = chain.invoke({
    'query': user_input,
    'data' : data
})

print(result)


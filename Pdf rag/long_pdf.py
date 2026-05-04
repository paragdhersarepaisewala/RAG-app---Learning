from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_mistralai import MistralAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os
load_dotenv()

doc = PyPDFLoader('PR.pdf')
data = doc.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 150
)

chunk = splitter.split_documents(data)

embedding = MistralAIEmbeddings(
    model='mistral-embed'
)

persist_dir = './pdf_rag'

if os.path.exists(persist_dir) and os.listdir(persist_dir):
    vector_store = Chroma(
        persist_directory=persist_dir,
        embedding_function=embedding
    )
else:
    vector_store = Chroma.from_documents(
        documents=chunk,
        embedding=embedding,
        persist_directory=persist_dir
    )
    vector_store.persist()

mmr_retriever = vector_store.as_retriever(
    search_type = 'mmr'
)

sst_ritriever = vector_store.as_retriever(
    search_type='similarity_score_threshold',
    search_kwargs={'score_threshold': 0.4,'k':5}
)

llm = ChatGoogleGenerativeAI(
    model = 'gemini-3.1-pro-preview'
)
system = '''You are a helpful assistent. Answer the user's query based on the context: {context}'''
history = []

prompt = ChatPromptTemplate.from_messages([
    ('system', system),
    MessagesPlaceholder('chat_history'),
    ('user', '{query}')
]
)

def get_history(x):
    return history

map_data = {
    'context': RunnablePassthrough() | mmr_retriever,
    'chat_history' : RunnableLambda(get_history),
    'query': RunnablePassthrough()
}

chain = map_data | prompt | llm | StrOutputParser()

while True:
    user_input = input("You: ")
    if user_input.lower() in ['quit', 'end', 'stop']:
        print("AI: Bye")
        break

    print("AI: ", end="", flush=True)
    full_response = ""
    
    # chain.stream returns strings because of StrOutputParser()
    for chunk in chain.stream(user_input):
        print(chunk, end="", flush=True)
        full_response += chunk
    
    # Update history after the stream finishes
    history.append(HumanMessage(content=user_input))
    history.append(AIMessage(content=full_response))
    print() # New line after stream ends
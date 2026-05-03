from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

history = []
load_dotenv()
embedding_model = HuggingFaceEmbeddings()

vector_db = Chroma(persist_directory="Web_rag", embedding_function=embedding_model)

retriever = vector_db.as_retriever(
    search_type='mmr',
    search_kwargs={'k': 3, 'fetch_k': 20}
)

llm = ChatGoogleGenerativeAI(
    model='gemini-3-flash-preview',
    temperature=0.3
)

system_prompt = """
Answer the question using ONLY the provided context.

- Ignore repeated or irrelevant information
- Extract only useful facts
- Be concise and structured

If the answer is not in the context, say:
"I don't know the answer."
Context:
{context}
"""

def get_history(x):
    return history[:5]
prompt = ChatPromptTemplate([
    ('system', system_prompt),
    MessagesPlaceholder('chat_history'),
    ('user','{query}')
]
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

map_data = {
    'context': retriever | RunnableLambda(format_docs),
    'chat_history': RunnableLambda(get_history),
    'query': RunnablePassthrough()
}

chain = map_data | prompt | llm | StrOutputParser()

while True:
    user = input("You: ")
    if user.lower() in ['end', 'quit']:
        break

    result = chain.invoke(user)

    print("----------------------")
    print("AI:" +result)

    history.append(HumanMessage(user))
    history.append(AIMessage(result))
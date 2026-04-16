from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv
load_dotenv()
llm = ChatMistralAI(
    model = 'mistral-small-2603',
    temperature=0.7
)

result = llm.invoke('Hello, Reply Hii')
print(result.content)
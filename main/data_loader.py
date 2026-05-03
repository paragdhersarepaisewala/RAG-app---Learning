from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()
url = "https://www.apple.com/in/iphone/"

doc = WebBaseLoader(url)
data = doc.load()
clean_data = re.sub(r'\s+', ' ',data[0].page_content).strip()
clean_data = re.sub(r'(Learn more|Buy|Compare all models|\\*‡.*)', '', clean_data)

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap = 100
)

chunks = splitter.split_text(clean_data)

embedding_model = HuggingFaceEmbeddings()

vector_db = Chroma.from_texts(chunks,embedding=embedding_model, persist_directory="Web_rag")

print("Database Created!")



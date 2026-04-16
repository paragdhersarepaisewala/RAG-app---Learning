from langchain_community.document_loaders import TextLoader
import json
data = TextLoader('document loaders/prompts.txt',encoding='utf-8')
doc = data.load()

# with open('Prompts.json','w') as f:
#     json.dump(doc.page_content(),f,indent=4)
print(doc[0].page_content)
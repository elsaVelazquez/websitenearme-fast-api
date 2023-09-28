# Load environment variables
from dotenv import load_dotenv,find_dotenv
load_dotenv(find_dotenv())

import os
import pinecone
from langchain.vectorstores import Pinecone

import pandas as pd


# connect to pinecone environment
pinecone.init(
    api_key=os.getenv('PINECONE_API_KEY'),  
    environment=os.getenv('PINECONE_ENV') 
)

# # Initialize Pinecone
INDEX_NAME = "websitenearme-fast-api"

index = pinecone.Index(INDEX_NAME)  # this index var is necessary for upsert later
pinecone.list_indexes()
pinecone.describe_index(INDEX_NAME)

import json
import requests
import os

data = {
    "prompt": "what are your prices?",
    "model": "text-embedding-ada-002"
}

openai = os.getenv('OPENAI_API_KEY')

response = requests.post("https://api.openai.com/v1/completions", json=data, headers={
    "Content-Type": "application/json",
    "Authorization": f"Bearer {openai}"
})

response_json = json.loads(response.text)

print(response_json["choices"][0]["text"])

# # from mock_embed import embed
# from mock_embed import embed_query


# import os
# import pinecone
# from langchain.vectorstores import Pinecone

# import pandas as pd
# from openai.embeddings_utils import get_embedding

# from langchain.vectorstores import Pinecone

# from scripts.data_pipeline_driver import INDEX_NAME

# # connect to pinecone environment
# pinecone.init(
#     api_key=os.getenv('PINECONE_API_KEY'),  
#     environment=os.getenv('PINECONE_ENV') 
# )

# # initialize vector store

# text_field = "text"

# # switch back to normal index for langchain
# index = pinecone.Index(INDEX_NAME)

# vectorstore = Pinecone(
#     index, embed_query, text_field
# )

# query = "prices?"


# vectorstore.similarity_search(
#     query,  # our search query
#     k=3  # return 3 most relevant docs
# )

# from langchain.chat_models import ChatOpenAI
# from langchain.chains import RetrievalQA

# # completion llm
# llm = ChatOpenAI(
#     openai_api_key=OPENAI_API_KEY,
#     model_name='gpt-3.5-turbo',
#     temperature=0.0
# )

# qa = RetrievalQA.from_chain_type(
#     llm=llm,
#     chain_type="stuff",
#     retriever=vectorstore.as_retriever()
# )

# # embedding = get_embedding(query)


# # # vectorstore = Pinecone(
# # #     index, embedding, text_field
# # # )

# # vectorstore = Pinecone(
# #     index, embed_query, text_field
# # )


# # vectorstore.similarity_search(
# #     query,  # our search query
# #     k=7  # return 3 most relevant docs
# # )
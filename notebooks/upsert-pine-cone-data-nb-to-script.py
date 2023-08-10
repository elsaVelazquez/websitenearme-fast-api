# %%
# Load environment variables
# python -m ipykernel install --user --name=webnearme-venv
from dotenv import load_dotenv,find_dotenv
load_dotenv(find_dotenv())



# %%
# allows notebook to work
import requests
from requests.packages.urllib3.util.ssl_ import create_urllib3_context

CIPHERS = (
    'ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM:DHE+CHACHA20:ECDH+AESGCM:ECDH+CHACHA20:DH+AESGCM:DH+CHACHA20:'
    'ECDHE+AES:!aNULL:!eNULL:!EXPORT:!DES:!MD5:!PSK:!RC4:!HMAC_SHA1:!SHA1:!DHE+AES:!ECDH+AES:!DH+AES'
)

requests.packages.urllib3.util.ssl_.DEFAULT_CIPHERS = CIPHERS
# Skip the following two lines if they cause errors
# requests.packages.urllib3.contrib.pyopenssl.DEFAULT_SSL_CIPHER_LIST = CIPHERS
# requests.packages.urllib3.contrib.pyopenssl.inject_into_urllib3()
requests.packages.urllib3.util.ssl_.create_default_context = create_urllib3_context


# %% [markdown]
# # we will create a hybrid index that can search semantic and keyword

# %%
# index = pinecone.Index('websitenearme-fast-api')

# %%
# this is our text corpus
all_sentences = ["Fast websites, fast!","HIRE US","MAIN PRODUCT/SERVICE 1","OUR EX1: WEBSITE DESIGN","","Initial website, same template as this site	1 design + hosting and domain (godaddy or transferred to godaddy)= ~$550","AI-powered SEO initial setup	$0, included","multilingual	$0, included","chatbot + maintenance	NOT INCLUDED","Website design","MAIN PRODUCT/SERVICE 2","OUR EX2: CHATBOTS FOR OUR WEBSITES","","Website + 1 site-specific chatbot	$100 * 1 time, chatbot setup, $20/month chatGPT subscription","Train model with AI-generated + your questions & answers	$100/ training round","Multiple chatbots with different personalities/ purposes	$100 each","Connect chatbot to your existing website (i.e., we did not build your website)	Please contact us with specifics, these start at $500/each and we reserve the right to say no","Chatbot types and prices","MAIN PRODUCT/SERVICE 3","EX3: SUBSCRIPTIONS","","monthly site health + maintenance + minor updates + 3 emergency updates	$50/month","monthly site health + maintenance + minor updates + 1 emergency update/week	$250/month","emergency update	$50","content generation- 4 posts, 1 per week	$100/month","Subscriptions and prices"]

# %%
# this uses HuggingFace 
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('flax-sentence-embeddings/all_datasets_v3_mpnet-base')

all_embeddings = model.encode(all_sentences)
print(all_embeddings.shape)
print(all_embeddings.shape[1]) # this is the dimension we need to instantiate our pinecone index
print(all_embeddings[0:1])

# %% [markdown]
# # the above is sufficient for upserting semantic searches, however the next step allows us to be able to do keyword searches

# %%
# this pip install allows notebooks to work
%pip install sacremoses
# this is HuggingFace transformer
from transformers import AutoTokenizer

# if using vast amounts of data, heed the parallelism warning

# transfo-xl tokenizer uses word-level encodings
tokenizer = AutoTokenizer.from_pretrained('transfo-xl-wt103')

all_tokens = [tokenizer.tokenize(sentence.lower()) for sentence in all_sentences]
all_tokens[0]

# %% [markdown]
# ## we will connect to pinecone

# %%
#Import and initialize Pinecone client

import os
import pinecone
from langchain.vectorstores import Pinecone

# connect to pinecone environment
pinecone.init(
    api_key=os.getenv('PINECONE_API_KEY'),  
    environment=os.getenv('PINECONE_ENV') 
)

# # Initialize Pinecone
INDEX_NAME = "websitenearme-fast-api"
DIMENSIONS=all_embeddings.shape[1]

# # Create and configure index if doesn't already exist
if INDEX_NAME not in pinecone.list_indexes():
    pinecone.create_index(
        name=INDEX_NAME, 
        metric="eucladian",
        dimension=DIMENSIONS
        )

index = pinecone.Index(INDEX_NAME)  # this index var is necessary for upsert later
pinecone.list_indexes()
pinecone.describe_index(INDEX_NAME)

# %% [markdown]
# ## now we will upsert the data

# %%
# only do this if i want a local json file
import json
from datetime import datetime

# Assume all_embeddings and all_tokens are already defined

# reformat the data
upserts = {'vectors': []}
for i, (embedding, tokens) in enumerate(zip(all_embeddings, all_tokens)):
    vector = {
        'id': f'{i}',
        'metadata': {
            'tokens': tokens,
            'time_stamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # Place timestamp within metadata
        },
        'values': embedding.tolist()
    }
    upserts['vectors'].append(vector)

# Add namespace at the root level
upserts['namespace'] = 'websitenearme'

# save to JSON
with open('./upsert.json', 'w') as f:
    json.dump(upserts, f, indent=4)



# %%
# upsert data straight into pinecone
import json
from datetime import datetime
from tqdm.auto import tqdm  # for progress bar

# Assume all_embeddings and all_tokens are already defined

# reformat the data
# data means the zipped embeddings and all tokens
# this correctly formats the json dump
# and is working to upload straight 
# into pincone as of Aguust 9, 2023
upserts = {'vectors': []}
for i, (embedding, tokens) in enumerate(zip(all_embeddings, all_tokens)):
    vector = {
        'id': f'{i}',
        'metadata': {
            'tokens': tokens,
            'time_stamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # Place timestamp within metadata
        },
        'values': embedding.tolist()
    }
    upserts['vectors'].append(vector)

# Add namespace at the root level
upserts['namespace'] = 'websitenearme'

# Use upserts['vectors'] as the dataset
dataset = upserts['vectors']

# Insert data as batches
batch_size = 100
for i in tqdm(range(0, len(dataset), batch_size)):
    # set end of current batch
    i_end = i + batch_size
    if i_end > len(dataset):
        # correct if batch is beyond dataset size
        i_end = len(dataset)
    batch = dataset[i: i_end]
    # Upsert the batch (assuming the structure of batch matches the expected format)
    index.upsert(vectors=batch)


# %%
# KEYWORD search
query_sentence = "prices"
xq = model.encode(query_sentence).tolist()
# get the data from Pinecone
result = index.query(xq, top_k=1, includeMetadata=True)
result

# %%
# SEMANTIC search
query_sentence = "what are your prices"
xq = model.encode(query_sentence).tolist()
# get the data from Pinecone
result = index.query(xq, top_k=1, includeMetadata=True)
result



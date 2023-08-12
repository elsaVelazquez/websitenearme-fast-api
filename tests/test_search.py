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

from sentence_transformers import SentenceTransformer

# this uses HuggingFace 
model = SentenceTransformer('flax-sentence-embeddings/all_datasets_v3_mpnet-base')

def dict_to_dataframe(data: dict) -> pd.DataFrame:
    """
    Convert the provided dictionary to a pandas DataFrame.
    
    Args:
    - data (dict): The dictionary containing token and metadata information.
    
    Returns:
    pd.DataFrame: The formatted DataFrame.
    """
    
    # Extract the 'matches' from the data
    matches = data['matches']
    
    # Prepare lists to store extracted data
    tokens_list, score_list, id_list = [], [], []

    # Iterate over matches and extract required data
    for match in matches:
        tokens = ', '.join(match['metadata']['tokens'])
        score = match['score']
        id = match['id']

        tokens_list.append(tokens)
        score_list.append(score)
        id_list.append(id)

    # Create a DataFrame
    df = pd.DataFrame({
        'Score': score_list,
        'ID': id_list,
        'Tokens': tokens_list
    })
    df_sorted = df.sort_values(by='Score', ascending=False, ignore_index=True)
    return df_sorted


def test_search(query_sentence: str, search_type: str, model, index, namespace:str) -> None:
    """
    Perform the search test based on the given search type.

    Args:
    - query_sentence (str): The sentence to query.
    - search_type (str): Type of search ("KEYWORD" or "SEMANTIC").
    - model: The model to encode the sentence.
    - index: The Pinecone index to query against.

    Returns:
    - None (prints out the results).
    """
    top_k_to_list = 15
    # Encode the query sentence
    xq = model.encode(query_sentence).tolist()

    # Get the data from Pinecone
    result = index.query(xq, top_k=top_k_to_list, includeMetadata=True, namespace=namespace)

    # Display results
    print(f"The row is enumerating the top_k={top_k_to_list} values")
    print(f"Query Sentence (*** the prompt ***): {query_sentence}\n")
    print("Results:\n")
    df = dict_to_dataframe(result)
    left_aligned_df = df.style.set_properties(**{'text-align': 'left'})
    print(left_aligned_df.to_string())

# Test the KEYWORD search
test_search("prices", "KEYWORD", model, index, "https_websitenearme")

# Test the SEMANTIC search
test_search("what are your prices", "SEMANTIC", model, index, "https_websitenearme")

# # test more namespaces
# Test the KEYWORD search
test_search("prompt engineer", "KEYWORD", model, index, "https_ai-architects")

# Test the SEMANTIC search
test_search("what do you do?", "SEMANTIC", model, index, "https_ai-architects")

#########################################
#### Ask questions using prompts ###############
import openai

limit = 3750


def retrieve(query):
    res = openai.Embedding.create(
        input=[query],
        engine=embed_model
    )

    # retrieve from Pinecone
    xq = res['data'][0]['embedding']

    # get relevant contexts
    res = index.query(xq, top_k=3, include_metadata=True)
    contexts = [
        x['metadata']['text'] for x in res['matches']
    ]

    # build our prompt with the retrieved contexts included
    prompt_start = (
        "Answer the question based on the context below.\n\n"+
        "Context:\n"
    )
    prompt_end = (
        f"\n\nQuestion: {query}\nAnswer:"
    )
    # append contexts until hitting limit
    for i in range(1, len(contexts)):
        if len("\n\n---\n\n".join(contexts[:i])) >= limit:
            prompt = (
                prompt_start +
                "\n\n---\n\n".join(contexts[:i-1]) +
                prompt_end
            )
            break
        elif i == len(contexts)-1:
            prompt = (
                prompt_start +
                "\n\n---\n\n".join(contexts) +
                prompt_end
            )
    return prompt
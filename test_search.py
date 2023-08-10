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
        'Tokens': tokens_list,
        'Score': score_list,
        'ID': id_list
    })

    return df


def test_search(query_sentence: str, search_type: str, model, index) -> None:
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
    # Encode the query sentence
    xq = model.encode(query_sentence).tolist()

    # Get the data from Pinecone
    result = index.query(xq, top_k=7, includeMetadata=True, namespace="https_websitenearme")

    # Display results
    print("The row is enumerating the top_k=3 value")
    print(f"Query Sentence: {query_sentence}\n")
    print("Results:\n")
    df = dict_to_dataframe(result)
    print(df)

# Test the KEYWORD search
test_search("prices", "KEYWORD", model, index)

# Test the SEMANTIC search
test_search("what are your prices", "SEMANTIC", model, index)

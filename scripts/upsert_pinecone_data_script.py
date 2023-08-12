import os
import requests
from datetime import datetime
import pinecone
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from tqdm.auto import tqdm
from typing import List, Tuple, Dict, Any
from tqdm.auto import tqdm
from time import sleep

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


def load_environment() -> None:
    """Load environment variables."""
    load_dotenv()


def configure_requests() -> None:
    """Configure requests to use specific ciphers."""
    CIPHERS = (
        'ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM:DHE+CHACHA20:ECDH+AESGCM:ECDH+CHACHA20:DH+AESGCM:DH+CHACHA20:'
        'ECDHE+AES:!aNULL:!eNULL:!EXPORT:!DES:!MD5:!PSK:!RC4:!HMAC_SHA1:!SHA1:!DHE+AES:!ECDH+AES:!DH+AES'
    )
    requests.packages.urllib3.util.ssl_.DEFAULT_CIPHERS = CIPHERS
    requests.packages.urllib3.util.ssl_.create_default_context = requests.packages.urllib3.util.ssl_.create_urllib3_context


import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter
from uuid import uuid4
from tqdm.auto import tqdm
tiktoken.encoding_for_model('gpt-3.5-turbo')
import openai
import pinecone

# upsert using tiktoken
def upsert_data_to_pinecone(index: Any, text: List[Dict[str, Any]], name_space) -> None:
    '''the dataset is same as text'''

    tokenizer = tiktoken.get_encoding('p50k_base')

    # create the length function
    def tiktoken_len(text):
        tokens = tokenizer.encode(
            text,
            disallowed_special=()
        )
        return len(tokens)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=20,
        length_function=tiktoken_len,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = []
    
    print(type(text))
    print(text[0])

    for record in tqdm(text):
        texts = text_splitter.split_text(record)
        chunks.extend([{
            'id': str(uuid4()),
            'text': texts[i],
            'chunk': i
        } for i in range(len(texts))])

    # initialize openai API key
    openai.api_key = os.getenv('OPENAI_API_KEY')  #platform.openai.com

    embed_model = "text-embedding-ada-002"

    # Rename the variable 'index' to 'index_name_input' to avoid confusion
    index_name_input = index

    # initialize connection to pinecone
    pinecone.init(
        api_key=os.getenv("PINECONE_API_KEY"),  # app.pinecone.io (console)
        environment=os.getenv("PINECONE_ENV")  # next to API key in console
    )

    # Get embeddings for the first text to determine the embedding dimension
    sample_res = openai.Embedding.create(input=[text[0]], engine=embed_model)
    embedding_dim = len(sample_res['data'][0]['embedding'])

    # check if index already exists (it shouldn't if this is first time)
    if index_name_input not in pinecone.list_indexes():
        # if does not exist, create index
        pinecone.create_index(
            index_name_input,
            dimension=embedding_dim,
            metric='dotproduct'
        )


    # Now that we have ensured the index exists, get a Pinecone index object to represent it
    pinecone_index = pinecone.GRPCIndex(index_name_input)

    # Use the 'pinecone' object to describe the index
    pinecone.describe_index(index_name_input)
    
    batch_size = 10  # how many embeddings we create and insert at once

    for i in tqdm(range(0, len(chunks), batch_size)):
        # find end of batch
        i_end = min(len(chunks), i+batch_size)
        meta_batch = chunks[i:i_end]
        # get ids
        ids_batch = [x['id'] for x in meta_batch]
        # get texts to encode
        texts = [x['text'] for x in meta_batch]
        # create embeddings (try-except added to avoid RateLimitError)
        try:
            res = openai.Embedding.create(input=texts, engine=embed_model)
            print(f'res.keys(): {res.keys()}')
            # print(f"res['data']: {res['data']}")
            print(f"len(res['data'][0]['embedding']), len(res['data'][1]['embedding']): {len(res['data'][0]['embedding']), len(res['data'][1]['embedding'])}")
            # import pdb; pdb.set_trace()
        except:
            done = False
            while not done:
                sleep(5)
                try:
                    res = openai.Embedding.create(input=texts, engine=embed_model)
                    done = True
                except:
                    pass
        embeds = [record['embedding'] for record in res['data']]
        # cleanup metadata
        meta_batch = [{
            'text': x['text'],
            'chunk': x['chunk'],
            'url': name_space
        } for x in meta_batch]
        to_upsert = list(zip(ids_batch, embeds, meta_batch))
        # upsert to Pinecone
        # index.upsert(vectors=to_upsert)
        # Use the 'pinecone_index' object to upsert the vectors
        pinecone_index.upsert(vectors=to_upsert, namespace=name_space)





def upsert_to_pinecone(sentences_file_path: str, name_space: str, INDEX_NAME: str) -> None:
    """
    Process the given sentences and upsert them to Pinecone.

    Args:
    - sentences_file_path: Path to the file containing sentences or lines.

    Returns:
    - None
    """
    print(f"upserting {name_space} to pinecone")
    load_environment()
    configure_requests()
    # all_sentences = create_all_sentences_lst(sentences_file_path)
    # all_embeddings = get_sentence_embeddings(all_sentences)
    # all_tokens = get_tokens(all_sentences)
    # index = connect_pinecone(INDEX_NAME)
    # upserts = prepare_data_for_upsert(all_embeddings, all_tokens, name_space)

    # upsert_data_to_pinecone(index, upserts['vectors'], name_space)
    with open(sentences_file_path, 'r') as file:
        data = file.readlines()
    
    upserts = upsert_data_to_pinecone(INDEX_NAME, data, name_space)


if __name__ == "__main__":
    upsert_to_pinecone(sentences_file_path, name_space, INDEX_NAME)

import os
import requests
from datetime import datetime
import pinecone
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from tqdm.auto import tqdm
from typing import List, Tuple, Dict, Any

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


def get_sentence_embeddings(sentences: List[str]) -> Tuple:
    """Retrieve sentence embeddings using SentenceTransformer."""
    model = SentenceTransformer('flax-sentence-embeddings/all_datasets_v3_mpnet-base')
    embeddings = model.encode(sentences)
    return embeddings


def get_tokens(sentences: List[str]) -> List[List[str]]:
    """Tokenize sentences using AutoTokenizer."""
    tokenizer = AutoTokenizer.from_pretrained('transfo-xl-wt103')
    tokens = [tokenizer.tokenize(sentence.lower()) for sentence in sentences]
    return tokens


def connect_pinecone(INDEX_NAME) -> Any:
    """Connect to Pinecone and return the index."""
    pinecone.init(
        api_key=os.getenv('PINECONE_API_KEY'),
        environment=os.getenv('PINECONE_ENV')
    )
    # INDEX_NAME = "websitenearme-fast-api"
    if INDEX_NAME not in pinecone.list_indexes():
        embeddings = get_sentence_embeddings(["dummy"])
        DIMENSIONS = embeddings.shape[1]
        pinecone.create_index(
            name=INDEX_NAME,
            metric="euclidean",
            dimension=DIMENSIONS
        )
    index = pinecone.Index(INDEX_NAME)
    return index


def prepare_data_for_upsert(all_embeddings: List[List[float]], all_tokens: List[List[str]], name_space: str) -> Dict[str, Any]:
    """Prepare data for upserting."""
    upserts = {'vectors': []}
    for i, (embedding, tokens) in enumerate(zip(all_embeddings, all_tokens)):
        vector = {
            'id': f'item_{i}',  # Changed id format to match your example
            'metadata': {
                'tokens': tokens,  # Assuming tokens is analogous to 'colors' in your example
                'time_stamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            },
            'values': embedding
        }
        upserts['vectors'].append(vector)
    upserts['namespace'] = name_space  # Set the namespace programmatically
    return upserts


def upsert_data_to_pinecone(index: Any, dataset: List[Dict[str, Any]], name_space) -> None:
    """Upsert data to Pinecone in batches."""
    # Setting ciphers only once for the entire session.
    # However, there are occasional errors when I remove it from the other places in the script
    # therefore leaving them for sake of reliability of code.
    CIPHERS = (
        'ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM:DHE+CHACHA20:ECDH+AESGCM:ECDH+CHACHA20:DH+AESGCM:DH+CHACHA20:'
        'ECDHE+AES:!aNULL:!eNULL:!EXPORT:!DES:!MD5:!PSK:!RC4:!HMAC_SHA1:!SHA1:!DHE+AES:!ECDH+AES:!DH+AES'
    )
    requests.packages.urllib3.util.ssl_.DEFAULT_CIPHERS = CIPHERS
    requests.packages.urllib3.util.ssl_.create_default_context = create_urllib3_context

    batch_size = 20
    for i in tqdm(range(0, len(dataset), batch_size)):
        i_end = i + batch_size
        if i_end > len(dataset):
            i_end = len(dataset)
        batch = dataset[i: i_end]
        print(f"Upserting batch with namespace: {name_space}")
        index.upsert(vectors=batch, namespace=name_space)


def create_all_sentences_lst(sentences_file_path: str) -> List[str]:
    """
    Extracts individual sentences or lines from the given file.
    
    Args:
    - sentences_file_path: Path to the file containing sentences or lines.

    Returns:
    - List of sentences or lines.
    """
    with open(sentences_file_path, 'r') as file:
        sentences = file.readlines()
    return [sentence.strip() for sentence in sentences if sentence.strip()]


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
    all_sentences = create_all_sentences_lst(sentences_file_path)
    all_embeddings = get_sentence_embeddings(all_sentences)
    all_tokens = get_tokens(all_sentences)
    index = connect_pinecone(INDEX_NAME)
    upserts = prepare_data_for_upsert(all_embeddings, all_tokens, name_space)

    upsert_data_to_pinecone(index, upserts['vectors'], name_space)


if __name__ == "__main__":
    upsert_to_pinecone(sentences_file_path, name_space, INDEX_NAME)
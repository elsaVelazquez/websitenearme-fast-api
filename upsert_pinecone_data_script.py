import os
import requests
from datetime import datetime
import pinecone
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from tqdm.auto import tqdm
from typing import List, Tuple, Dict, Any


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


def connect_pinecone() -> Any:
    """Connect to Pinecone and return the index."""
    pinecone.init(
        api_key=os.getenv('PINECONE_API_KEY'),
        environment=os.getenv('PINECONE_ENV')
    )
    INDEX_NAME = "websitenearme-fast-api"
    if INDEX_NAME not in pinecone.list_indexes():
        embeddings = get_sentence_embeddings(["dummy"])
        DIMENSIONS = embeddings.shape[1]
        pinecone.create_index(
            name=INDEX_NAME,
            metric="eucladian",
            dimension=DIMENSIONS
        )
    index = pinecone.Index(INDEX_NAME)
    return index


def prepare_data_for_upsert(all_embeddings: List[List[float]], all_tokens: List[List[str]]) -> Dict[str, Any]:
    """Prepare data for upserting."""
    upserts = {'vectors': []}
    for i, (embedding, tokens) in enumerate(zip(all_embeddings, all_tokens)):
        vector = {
            'id': f'{i}',
            'metadata': {
                'tokens': tokens,
                'time_stamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            },
            'values': embedding
        }
        upserts['vectors'].append(vector)
    upserts['namespace'] = 'websitenearme'
    return upserts


def upsert_data_to_pinecone(index: Any, dataset: List[Dict[str, Any]]) -> None:
    """Upsert data to Pinecone in batches."""
    batch_size = 100
    for i in tqdm(range(0, len(dataset), batch_size)):
        i_end = i + batch_size
        if i_end > len(dataset):
            i_end = len(dataset)
        batch = dataset[i: i_end]
        index.upsert(vectors=batch)


def main():
    load_environment()
    configure_requests()
    all_sentences = [
        "Fast websites, fast!", "HIRE US", "MAIN PRODUCT/SERVICE 1", "OUR EX1: WEBSITE DESIGN", "",
        # ... [Truncated for brevity] ...
    ]

    all_embeddings = get_sentence_embeddings(all_sentences)
    all_tokens = get_tokens(all_sentences)
    index = connect_pinecone()
    upserts = prepare_data_for_upsert(all_embeddings, all_tokens)
    upsert_data_to_pinecone(index, upserts['vectors'])


if __name__ == "__main__":
    main()

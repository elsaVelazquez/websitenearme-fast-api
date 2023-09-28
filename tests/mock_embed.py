# mock_embed.py

from openai.embeddings_utils import get_embedding

def embed_query(query):
    return get_embedding(query, model='text-embedding-ada-002')

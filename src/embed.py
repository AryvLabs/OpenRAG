from ingestion import load_corpus
from sentence_transformers import SentenceTransformer
import chromadb
import os
import sys

filetype = sys.argv[1]
db_path = os.getcwd() + "/db"
chroma_client = chromadb.PersistentClient(path=db_path)
db_collection = chroma_client.get_or_create_collection(name="RAG_Data")

aiEmbeddingModel = SentenceTransformer('all-MiniLM-L6-v2')
corpus = load_corpus(filetype)

hfEmbedding = aiEmbeddingModel.encode(corpus).tolist()
idList = [f"id_{i}" for i in range(len(hfEmbedding))]
print (idList)

db_collection.add(
    embeddings = hfEmbedding,
    documents = corpus,
    ids=idList
)


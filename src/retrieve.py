import chromadb
import os
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
import numpy as np

# Load models once at module level — avoids reloading on every query
_embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
_cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')


def reciprocal_rank_fusion(dense_results, sparse_results, k=60):
    """
    Fuse dense (vector) and sparse (BM25) results using RRF.
    Returns a list of document IDs sorted by fused score.
    """
    scores = {}

    for rank, doc_id in enumerate(dense_results):
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)

    for rank, doc_id in enumerate(sparse_results):
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)

    return sorted(scores.keys(), key=lambda x: scores[x], reverse=True)


def retrieve_embedding(user_input, n_results=3):
    """
    Hybrid retrieval combining dense vector search and BM25 sparse search,
    followed by cross-encoder reranking.
    Returns top n_results documents.
    """
    db_path = os.getcwd() + "/src/db"
    chroma_client = chromadb.PersistentClient(path=db_path)
    db_collection = chroma_client.get_or_create_collection(name="RAG_Data")

    # Embed the user query using the cached model
    query_embedding = _embedding_model.encode(user_input).tolist()

    # Dense retrieval: vector similarity search
    dense_results = db_collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results * 2
    )
    dense_ids = dense_results["ids"][0] if dense_results["ids"] else []

    # Sparse retrieval: BM25 keyword search over all stored documents
    all_docs = db_collection.get(include=["documents"])
    corpus_docs = all_docs["documents"]
    corpus_ids = all_docs["ids"]

    # Guard: if DB is empty, return early
    if not corpus_docs:
        return []

    tokenized_corpus = [doc.lower().split() for doc in corpus_docs]
    tokenized_query = user_input.lower().split()

    bm25 = BM25Okapi(tokenized_corpus)
    bm25_scores = bm25.get_scores(tokenized_query)

    top_bm25_indices = np.argsort(bm25_scores)[::-1][:n_results * 2]
    sparse_ids = [corpus_ids[i] for i in top_bm25_indices]

    # Fuse results using RRF
    fused_ids = reciprocal_rank_fusion(dense_ids, sparse_ids)

    # Guard: if fusion returned nothing, return early
    if not fused_ids:
        return []

    # Retrieve candidate documents for reranking
    candidate_ids = fused_ids[:n_results * 2]
    candidate_docs = db_collection.get(
        ids=candidate_ids,
        include=["documents"]
    )

    # Guard: if no candidate documents were found, return early
    if not candidate_docs["documents"]:
        return []

    # Rerank using the cached cross-encoder
    pairs = [[user_input, doc] for doc in candidate_docs["documents"]]
    rerank_scores = _cross_encoder.predict(pairs)

    ranked = sorted(
        zip(rerank_scores, candidate_docs["documents"]),
        key=lambda x: x[0],
        reverse=True
    )

    return [doc for _, doc in ranked[:n_results]]

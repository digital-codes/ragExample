import argparse 
import os
import json
import numpy as np
from joblib import Parallel, delayed

# embedder will complain on tokenizer parallelism
# set TOKENIZERS_PARALLELISM to False before running
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../rag')))
from ragInstrumentation import measure_execution_time

DEBUG = True

@measure_execution_time
def load_vectors(filename):
    """Load all vectors from a binary file of float32, shape: (N, args.dim)."""
    file_size = os.path.getsize(filename)
    bytes_per_record = args.dim * 4  # float32 is 4 bytes
    if file_size % bytes_per_record != 0:
        raise ValueError("File size not divisible by record size. Invalid file?")

    num_records = file_size // bytes_per_record
    print(f"Number of vectors: {num_records}")

    # Read all bytes
    with open(filename, 'rb') as f:
        raw = f.read()

    # Convert to float32 array
    arr = np.frombuffer(raw, dtype=np.float32)
    # Reshape to [N, args.dim]
    arr = arr.reshape(num_records, args.dim)

    # Normalize each vector to unit length.
    # Axis=1 means we take the norm across each row (each vector).
    norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9
    arr_normalized = arr / norms
    
    return arr_normalized


#########################
def compute_cosine_similarity(query: np.ndarray, doc: np.ndarray) -> float:
    """
    Compute the cosine similarity between two vectors.
    """
    # np.dot is efficient for dot products
    # Use a small epsilon check to avoid dividing by zero
    denom = (np.linalg.norm(query) * np.linalg.norm(doc)) + 1e-9
    return np.dot(query, doc) / denom

def parallel_compute_similarities(query: np.ndarray, vectors: np.ndarray, n_jobs=-1):
    """
    Compute cosine similarities between a `query` vector and an array of `vectors`
    in parallel. 
    Returns a list of similarities and the corresponding sorted indices 
    (descending similarity).
    
    :param query: 1D numpy array representing the query vector
    :param vectors: 2D numpy array of shape (N, D), where N is the number
                    of vectors and D is their dimension
    :param n_jobs: number of parallel workers; -1 uses all cores
    """
    # Ensure query is a 1D array
    query = query.flatten()
    
    # Parallel computation
    similarities = Parallel(n_jobs=n_jobs)(
        delayed(compute_cosine_similarity)(query, v) 
        for v in vectors
    )
    
    # Sort by similarity descending
    sorted_indices = np.argsort(similarities)[::-1]
    sorted_sims = [similarities[i] for i in sorted_indices]
    
    return sorted_sims, sorted_indices

        
@measure_execution_time
def query_vectors(vectors, query, num_neighbors=5, data=None):
    query_norm = np.linalg.norm(query) + 1e-9
    query_normalized = query / query_norm

    distances, indices = parallel_compute_similarities(query_normalized, vectors, n_jobs=-1)
    
    return indices[:num_neighbors], distances[:num_neighbors]

@measure_execution_time
def get_top_n_dedup(similarities, indices, identifiers, top_n):
    """
    We have similarities sorted descending, plus matching indices.
    We walk through them in descending similarity order. For each doc_id,
    the *first* time we see it, that is its best (maximum) similarity.
    
    We keep track of unique doc_ids. Once we reach 'top_n' unique doc_ids,
    we break and return our results.
    
    :param similarities: 1D array (descending) of shape (M,)
    :param indices:      1D array of shape (M,), matching 'similarities'
    :param identifiers:  array or function mapping row index -> doc_id
    :param top_n:        how many unique doc_ids we want to collect
    
    :return: A list of (doc_id, similarity) of length <= top_n,
             in descending similarity order.
    """
    deduped_ids = set()
    results = []
    
    count_unique = 0
    for i in range(len(similarities)):
        sim = similarities[i]
        idx = indices[i]
        doc_id = identifiers[idx]  # e.g. if identifiers is a numpy array or dict
        
        # If we've not seen this doc_id yet, this is its best similarity
        if doc_id not in deduped_ids:
            deduped_ids.add(doc_id)
            results.append((doc_id, sim))
            count_unique += 1
            
            # If we've reached top_n unique doc_ids, we can stop
            if count_unique >= top_n:
                break
    
    # 'results' is already in descending similarity order
    return results

    # # We want the top 5 unique doc_ids
    # top_n = 5
    # dedup_results = single_pass_top_n_dedup(similarities, indices, identifiers, top_n)
    
    # print(f"Found {len(dedup_results)} unique doc_ids (up to {top_n}):")
    # for doc_id, sim in dedup_results:
    #     print(f"doc_id={doc_id}, similarity={sim:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--command',choices=["search","text"])      # option that takes a value
    parser.add_argument('-v', '--vectors', default="vectors.bin")      # option that takes a value
    parser.add_argument('-q', '--query')      # option that takes a value
    parser.add_argument('-i', '--items', type=int, default=5)      # option that takes a value
    parser.add_argument('-d', '--dim', type=int, default=1024)      # option that takes a value
    args = parser.parse_args()

    if args.command == "search":
        if not os.path.exists(args.vectors):
            print(f"Source file {args.vectors} not found.")
            parser.print_help()
            exit(1)
        vectors = load_vectors(args.vectors)
        if args.query == None:
            print(f"No query.")
            parser.print_help()
            exit(1)
        query = json.loads(args.query)
        query_vec = np.array(query, dtype=np.float32).reshape(1, args.dim)
        result, dists = query_vectors(vectors,query_vec[0], args.items)
        print("NN indices:", result)
        print("NN distances:", dists)
    elif args.command == "text":
        if not os.path.exists(args.vectors):
            print(f"Source file {args.vectors} not found.")
            parser.print_help()
            exit(1)
        vectors = load_vectors(args.vectors)
        if args.query == None:
            print(f"No query.")
            parser.print_help()
            exit(1)
        query_text = args.query
        try:
            import ragDeployUtils as rag
            embedder = rag.Embedder(provider="local")
        except:
            print("No local embedder available")
            embedder = rag.Embedder()
        query = embedder.encode(query_text)["data"][0]["embedding"]
        query_vec = np.array(query, dtype=np.float32).reshape(1, args.dim)
        result, dists = query_vectors(vectors,query_vec[0], args.items)
        print("NN indices:", result)
        print("NN distances:", dists)
    else:
        parser.print_help()
        exit(1)


import argparse 
import os
import json
import numpy as np
from joblib import Parallel, delayed

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../rag')))
from ragInstrumentation import measure_execution_time

DEBUG = False

@measure_execution_time
def load_vectors(filename,dim):
    """
    Load and normalize vectors from a binary file.
    Args:
        filename (str): Path to the binary file containing float32 vectors.
    Returns:
        np.ndarray: A 2D numpy array of shape (N, args.dim) containing the normalized vectors.
    Raises:
        ValueError: If the file size is not divisible by the size of a single vector record.
    Notes:
        - The binary file is expected to contain vectors of shape (N, args.dim) where each element is a float32.
        - Each vector is normalized to unit length.
    """
    file_size = os.path.getsize(filename)
    bytes_per_record = dim * 4  # float32 is 4 bytes
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
    arr = arr.reshape(num_records, dim)

    # Normalize each vector to unit length.
    # Axis=1 means we take the norm across each row (each vector).
    norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9
    arr_normalized = arr / norms
    
    return arr_normalized


#########################
def compute_cosine_similarity(query: np.ndarray, doc: np.ndarray) -> float:
    """
        Parameters:
        query (np.ndarray): The first vector.
        doc (np.ndarray): The second vector.

        Returns:
        float: The cosine similarity between the two vectors.
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
def query_vectors(vectors, query, num_neighbors=5):
    """
    Find the nearest neighbors to a query vector from a set of vectors.
    Parameters:
    vectors (numpy.ndarray): A 2D array where each row is a vector from the dataset.
    query (numpy.ndarray): A 1D array representing the query vector.
    num_neighbors (int, optional): The number of nearest neighbors to return. Default is 5.
    Returns:
    tuple: A tuple containing two elements:
        - indices (numpy.ndarray): The indices of the nearest neighbors in the dataset.
        - distances (numpy.ndarray): The distances of the nearest neighbors from the query vector.
    """
    query_norm = np.linalg.norm(query) + 1e-9
    query_normalized = query / query_norm

    distances, indices = parallel_compute_similarities(query_normalized, vectors, n_jobs=-1)
    
    if num_neighbors > 0:
        return indices[:num_neighbors], distances[:num_neighbors]
    else:
        return indices, distances

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--vectors', default="vectors.bin")      # option that takes a value
    parser.add_argument('-q', '--query')      # option that takes a value
    parser.add_argument('-i', '--items', type=int, default=5)      # option that takes a value
    parser.add_argument('-d', '--dim', type=int, default=1024)      # option that takes a value
    args = parser.parse_args()

    if not os.path.exists(args.vectors):
        print(f"Source file {args.vectors} not found.")
        parser.print_help()
        exit(1)
    vectors = load_vectors(args.vectors,args.dim)
    if args.query == None:
        print(f"No query.")
        parser.print_help()
        exit(1)
    query = json.loads(args.query)
    query_vec = np.array(query, dtype=np.float32).reshape(1, args.dim)
    result, dists = query_vectors(vectors,query_vec[0], args.items)
    print("NN indices:", result)
    print("NN distances:", dists)


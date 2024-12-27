import faiss
import argparse 
import os
import json
import numpy as np
from ragInstrumentation import measure_execution_time


DIM = 384
DEBUG = False

@measure_execution_time
def load_vectors(filename):
    """Load all vectors from a binary file of float32, shape: (N, DIM)."""
    file_size = os.path.getsize(filename)
    bytes_per_record = DIM * 4  # float32 is 4 bytes
    if file_size % bytes_per_record != 0:
        raise ValueError("File size not divisible by record size. Invalid file?")

    num_records = file_size // bytes_per_record
    print(f"Number of vectors: {num_records}")

    # Read all bytes
    with open(filename, 'rb') as f:
        raw = f.read()

    # Convert to float32 array
    arr = np.frombuffer(raw, dtype=np.float32)
    # Reshape to [N, DIM]
    arr = arr.reshape(num_records, DIM)

    # Normalize each vector to unit length.
    # Axis=1 means we take the norm across each row (each vector).
    norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9
    arr_normalized = arr / norms
    
    return arr_normalized


###############################################################################
# 1) Build Index
###############################################################################
@measure_execution_time
def build_index(data, metric='euclidean'):
    """
    Build a Faiss index from a numpy array of shape (N, D).
    
    :param data:   numpy array, shape (N, D)
    :param metric: str, either 'euclidean' or 'angular' (cosine-like)
    :return:       a Faiss index object
    """
    N, D = data.shape
    
    # Convert to float32 if not already
    if data.dtype != np.float32:
        data = data.astype(np.float32)

    if metric == 'euclidean':
        # Use L2 distance
        index = faiss.IndexFlatL2(D)
        index.add(data)
    elif metric == 'angular':
        # For cosine similarity in Faiss: 
        # 1) Normalize each vector to unit length
        # 2) Use an IndexFlatIP (inner product) 
        norms = np.linalg.norm(data, axis=1, keepdims=True) + 1e-9
        data_normed = data / norms
        index = faiss.IndexFlatIP(D)
        index.add(data_normed)
    else:
        raise ValueError("metric must be either 'euclidean' or 'angular'")

    return index


###############################################################################
# 2) Save Index
###############################################################################
@measure_execution_time
def save_index(index, filename):
    """
    Saves the Faiss index to a file.
    """
    faiss.write_index(index, filename)


###############################################################################
# 3) Load Index
###############################################################################
@measure_execution_time
def load_index(filename):
    """
    Loads an existing Faiss index from disk.
    
    :return: a Faiss index object
    """
    index = faiss.read_index(filename)
    return index


###############################################################################
# 4) Query Index
###############################################################################
@measure_execution_time
def query_index(index, query, num_neighbors=5, metric='euclidean'):
    """
    Query the Faiss index for nearest neighbors of a given vector.
    
    :param index:        a Faiss index (already built or loaded)
    :param query:        1D numpy array, shape (D,)
    :param num_neighbors number of neighbors to retrieve
    :param metric:       'euclidean' or 'angular', consistent with build_index
    :return: (indices, distances)
             - indices:   list of neighbor IDs
             - distances: list of distances or similarities 
                          (depending on metric & index type)
    """
    # Ensure query is float32 and shaped (1, D)
    query = query.astype(np.float32).reshape(1, -1)

    # If using "angular", then the index expects normalized vectors
    # so we also normalize the query before searching the IP index.
    if metric == 'angular':
        norm_q = np.linalg.norm(query, axis=1, keepdims=True) + 1e-9
        query_normed = query / norm_q
        distances, indices = index.search(query_normed, num_neighbors)
        # distances are "inner product" values (the higher, the more similar).
        # Actually, IndexFlatIP returns *dot products*, so bigger => closer
        # but if you want to interpret them as "distance," it's reversed logic.
    else:
        # metric='euclidean' => we assume an IndexFlatL2
        distances, indices = index.search(query, num_neighbors)
        # distances are L2-squared, i.e. the sum of squared differences, 
        # or L2 if Faiss is configured that way (IndexFlatL2 typically 
        # returns the sum of squared differences).
    
    # The shapes of distances, indices will be (1, num_neighbors)
    distances = distances[0]
    indices = indices[0]
    return indices, distances


###############################################################################
# Example Usage (if you run this file directly)
###############################################################################

    # # Suppose we want Euclidean index
    # index = build_index(data, metric='euclidean')
    
    # # Save
    # save_index(index, "faiss_index.bin")
    
    # # Load
    # loaded_index = load_index("faiss_index.bin")
    
    # # Query
    # query_vec = np.random.randn(D).astype(np.float32)
    # k = 5
    # nn_ids, nn_dists = query_index(loaded_index, query_vec, k, metric='euclidean')
    
    # print(f"Nearest Neighbors IDs: {nn_ids}")
    # print(f"Dists (L2-squared): {nn_dists}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--command',choices=["build","load","search","text"])      # option that takes a value
    parser.add_argument('-v', '--vectors', default="vectors.bin")      # option that takes a value
    parser.add_argument('-i', '--index',default="index_a.bin")      # option that takes a value
    parser.add_argument('-q', '--query')      # option that takes a value
    args = parser.parse_args()

    if args.command == "build":
        if not os.path.exists(args.vectors):
            print(f"Source file {args.vectors} not found.")
            parser.print_help()
            exit(1)
        vectors = load_vectors(args.vectors)
        index = build_index(vectors)
        save_index(index, args.index)
    elif args.command == "load":
        if not os.path.exists(args.index):
            print(f"Source file {args.index} not found.")
            parser.print_help()
            exit(1)
        index = load_index(args.index)
    elif args.command == "search":
        if not os.path.exists(args.index):
            print(f"Source file {args.index} not found.")
            parser.print_help()
            exit(1)
        if args.query == None:
            print(f"No query.")
            parser.print_help()
            exit(1)
        index = load_index(args.index)
        # Create a dummy query vector
        #query_vec = np.ones((1, DIM), dtype=np.float32)
        query = json.loads(args.query)
        query_vec = np.array(query, dtype=np.float32).reshape(1, DIM)
        k = 5
        result, dists = query_index(index,query_vec[0], k)
        print("NN indices:", result)
        print("NN distances:", dists)
    elif args.command == "text":
        if not os.path.exists(args.index):
            print(f"Source file {args.index} not found.")
            parser.print_help()
            exit(1)
        if args.query == None:
            print(f"No query.")
            parser.print_help()
            exit(1)
        index = load_index(args.index)
        # Create a dummy query vector
        #query_vec = np.ones((1, DIM), dtype=np.float32)
        query_text = args.query
        try:
            import ragDeployUtils as rag
            embedder = rag.Embedder(provider="local")
        except:
            print("No local embedder available")
            embedder = rag.Embedder()
        query = embedder.encode(query_text)["data"][0]["embedding"]
        query_vec = np.array(query, dtype=np.float32).reshape(1, DIM)
        k = 5
        result, dists = query_index(index,query_vec[0], k)
        print("NN indices:", result)
        print("NN distances:", dists)
    else:
        parser.print_help()
        exit(1)


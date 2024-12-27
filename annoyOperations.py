from annoy import AnnoyIndex
import argparse 
import os
import json
import numpy as np

DIM = 384

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


# Notes
#     Annoy requires you to manually add_item(i, vector).
#     metric='angular' is typically used for cosine-like similarity. Annoy’s “angular distance” = 2 * (1 - cos(\theta)).
#     If you want pure L2 distance, use metric='euclidean'.
def build_index(data, metric='euclidean', n_trees=8):
    """
    Build an Annoy index from a numpy array of shape (N, D).
    
    :param data:    numpy array, shape (N, D)
    :param metric:  str, either 'angular' (cosine-like) or 'euclidean'
    :param n_trees: number of trees to build (higher => better accuracy, slower build)
    :return:        an AnnoyIndex object
    """
    N, D = data.shape
    
    # Create the Annoy index specifying the dimension D and metric
    index = AnnoyIndex(D, metric)
    
    # Add each vector to the index
    for i in range(N):
        index.add_item(i, data[i])
    
    # Build the index
    index.build(n_trees)
    
    return index


def save_index(index, filename):
    """
    Saves the Annoy index to a file.
    """
    index.save(filename)
    

def load_index(filename, dimension=DIM, metric='euclidean'):
    """
    Loads an existing Annoy index from disk.
    
    :param filename:  path to the saved index
    :param dimension: dimension (D) used when building
    :param metric:    same metric as used for building the index
    :return:          an AnnoyIndex object
    """
    index = AnnoyIndex(dimension, metric)
    index.load(filename)
    return index
        
def query_index(index, query, num_neighbors=5, data=None):
    """
    Query the Annoy index for nearest neighbors of a given vector.
    
    :param index:         an AnnoyIndex (already built or loaded)
    :param query_vector:  1D numpy array, shape (D,)
    :param num_neighbors: number of neighbors to retrieve
    :param data:          optional numpy array of shape (N, D) 
                          if you want to look up actual vectors or re-check distances
    :return: 
        (indices, distances)
        - indices:   list of neighbor IDs
        - distances: list of distances (Annoy can return them if include_distances=True)
    """
    query_norm = np.linalg.norm(query) + 1e-9
    query_normalized = query / query_norm
    # Annoy can return the distances if we ask for it:
    indices, distances = index.get_nns_by_vector(
        query_normalized,
        num_neighbors,
        include_distances=True
    )
    
    # Distances are sorted ascending by default (closest neighbor first).
    # If 'metric=angular', these distances are "angular distance" = 2 * (1 - cos_sim).
    # If 'metric=euclidean', these are L2 distances.
    
    return indices, distances



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--command',choices=["build","load","search"])      # option that takes a value
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

    else:
        parser.print_help()
        exit(1)


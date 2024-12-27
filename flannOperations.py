from pyflann import FLANN
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


def build_index(data):
    flann = FLANN()
    flann.build_index(data, algorithm="kdtree", trees=8)
    return flann

def save_index(flann, filename):
    flann.save_index(filename)
    
def load_index(filename, data):
    flann = FLANN()
    flann.load_index(filename, data)
    return flann

def query_index(flann, query, num_neighbors=5):
    query_norm = np.linalg.norm(query) + 1e-9
    query_normalized = query / query_norm
    indices, dists = flann.nn_index(query_normalized, num_neighbors)
    # 'dists' often are squared L2 distances. Let's convert:
    distances = np.sqrt(dists)  # Now it's actual L2
    # already sorted with smallest distance first
    return indices, distances


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--command',choices=["build","load","search"])      # option that takes a value
    parser.add_argument('-v', '--vectors', default="vectors.bin")      # option that takes a value
    parser.add_argument('-i', '--index',default="index.bin")      # option that takes a value
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
        vectors = load_vectors(args.vectors)
        index = load_index(args.index, vectors)
    elif args.command == "search":
        if not os.path.exists(args.vectors):
            print(f"Source file {args.vectors} not found.")
            parser.print_help()
            exit(1)
        if not os.path.exists(args.index):
            print(f"Source file {args.index} not found.")
            parser.print_help()
            exit(1)
        if args.query == None:
            print(f"No query.")
            parser.print_help()
            exit(1)
        vectors = load_vectors(args.vectors)
        index = load_index(args.index, vectors)
        # Create a dummy query vector
        #query_vec = np.ones((1, DIM), dtype=np.float32)
        query = json.loads(args.query)
        query_vec = np.array(query, dtype=np.float32).reshape(1, DIM)
        k = 5
        result, dists = query_index(index,query_vec, k)
        print("NN indices:", result)
        print("NN distances:", dists)

    else:
        parser.print_help()
        exit(1)


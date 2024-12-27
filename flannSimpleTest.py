import numpy as np
from pyflann import FLANN

DIM = 384  # or 768
N = 100000  # e.g. 100k, can go up to 1 million if RAM allows

def generate_data(num_vectors, dim):
    # Return a (num_vectors, dim) array of float32
    data = np.zeros((num_vectors, dim), dtype=np.float32)
    for i in range(num_vectors):
        for j in range(dim):
            data[i, j] = i + 0.001 * j
    return data

if __name__ == "__main__":
    # 1) Generate or load data
    dataset = generate_data(N, DIM)

    flann = FLANN()
    # 2) Build index with KD-tree approach
    index_params = flann.build_index(dataset, algorithm='kdtree', trees=8)
    print("FLANN index built with params:", index_params)

    # 3) Query
    query_vec = np.ones((1, DIM), dtype=np.float32)  # shape (1, 384)
    k = 5
    result, dists = flann.nn_index(query_vec, k)
    print("Nearest neighbor indices:", result)
    print("Distances:", dists)


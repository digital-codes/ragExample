import numpy as np
import json

DIM = 384  # or 768
N = 100  # e.g. 100k, can go up to 1 million if RAM allows

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
    # 2) Write dataset to file
    with open('vectors2.bin', 'wb') as f:
        f.write(dataset.tobytes())
    v = dataset[0]
    print(json.dumps(list(v.astype(float))))
    
    
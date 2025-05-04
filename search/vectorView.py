import numpy as np
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.express as px
import requests
import argparse
import os

def load_vectors(path, dim):
    """
    Load and normalize vectors from a binary file.
    Args:
        path (str): Path to the binary file containing float32 vectors.
    Returns:
        np.ndarray: A 2D numpy array of shape (N, args.dim) containing the normalized vectors.
    Raises:
        ValueError: If the file size is not divisible by the size of a single vector record.
    Notes:
        - The binary file is expected to contain vectors of shape (N, args.dim) where each element is a float32.
        - Each vector is normalized to unit length.
    """
    file_size = os.path.getsize(path)
    bytes_per_record = dim * 4  # float32 is 4 bytes
    if file_size % bytes_per_record != 0:
        raise ValueError("File size not divisible by record size. Invalid file?")
        return None

    num_records = file_size // bytes_per_record
    print(f"Number of vectors: {num_records}")

    # Read all bytes
    with open(path, 'rb') as f:
        raw = f.read()

    # Convert to float32 array
    arr = np.frombuffer(raw, dtype=np.float32)
    # Reshape to [N, args.dim]
    arr = arr.reshape(num_records, dim)
    N,D = arr.shape
    print(f"Loaded {N} vectors with dim {D}")

    return arr

def reduce_dimensions(vectors, method='umap'):
    n_samples = len(vectors)
    scale = 1000 if n_samples > 10000 else 100 if n_samples > 1000 else 10
    print(f"Running {method.upper()} on {n_samples} vectors")

    if method == 'umap':
        reducer = umap.UMAP(n_neighbors=min(15, n_samples // scale), min_dist=0.1, metric='cosine')
    elif method == 'tsne':
        reducer = TSNE(n_components=2, perplexity=min(30, n_samples // scale), n_iter= max(250,scale), verbose=1)
    elif method == 'pca':
        reducer = PCA(n_components=2)
    else:
        raise ValueError(f"Unsupported method: {method}")
    
    return reducer.fit_transform(vectors)

def plot_embedding(embedding, title="Vector Space Visualization"):
    fig = px.scatter(x=embedding[:, 0], y=embedding[:, 1], title=title)
    fig.update_layout(xaxis_title='X', yaxis_title='Y')
    fig.show()

def query_neighbors(vector, search_url, top_k=5):
    payload = {
        "vector": vector.tolist(),
        "top_k": top_k
    }
    try:
        response = requests.post(search_url, json=payload, timeout=2)
        return response.json()
    except Exception as e:
        print(f"Search error: {e}")
        return {}

def main():
    parser = argparse.ArgumentParser(description="Visualize vector embeddings with UMAP/t-SNE/PCA.")
    parser.add_argument("vector_file", help="Path to binary file with float32 vectors")
    parser.add_argument("--method", default="umap", choices=["umap", "tsne", "pca"], help="Dimensionality reduction method")
    parser.add_argument("--dim", type=int, default=1024, help="Dimension")
    parser.add_argument("--search_url", type=str, help="(Optional) HTTP endpoint for vector search")
    parser.add_argument("--query_test", action="store_true", help="Run a test query on first 3 vectors")

    args = parser.parse_args()

    vectors = load_vectors(args.vector_file, args.dim)
    embedding = reduce_dimensions(vectors, args.method)
    plot_embedding(embedding, title=f"{args.method.upper()} Projection")

    if args.search_url and args.query_test:
        print("\nTesting search service with first 3 vectors:")
        for i in range(min(3, len(vectors))):
            print(f"\nQuerying vector {i}")
            result = query_neighbors(vectors[i], args.search_url)
            print(result)

if __name__ == "__main__":
    main()

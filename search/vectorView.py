import numpy as np
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.express as px
import requests
import argparse
import os
import hdbscan
import pandas as pd

# umap on karis titles (28000) results in 312 clusters.

SEED = 42

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
        reducer = umap.UMAP(n_neighbors=min(50, n_samples // scale), min_dist=0.1, 
                            metric='cosine', n_jobs = 1, random_state=SEED, init="pca" )
    elif method == 'tsne':
        #reducer = TSNE(n_components=2, perplexity=min(15, 2*n_samples // scale), n_iter= max(250,scale), verbose=1)
        reducer = TSNE(random_state=SEED,n_components=2, perplexity=100, n_iter= 1000, verbose=1)
    elif method == 'pca':
        reducer = PCA(random_state=SEED,n_components=2)
    else:
        raise ValueError(f"Unsupported method: {method}")
    
    return reducer.fit_transform(vectors)

def find_clusters(embedding, min_size=20):
    print("Finding clusters with HDBSCAN...")
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_size, prediction_data=True,core_dist_n_jobs=1) # error at end of program. ignore
    labels = clusterer.fit_predict(embedding)
    return labels


def plot_embedding(embedding, labels, indices, title="Vector Space Visualization"):
    import pandas as pd

    df = pd.DataFrame({
        'x': embedding[:, 0],
        'y': embedding[:, 1],
        'cluster': labels,
        'index': indices
    })

    fig = px.scatter(
        df, x='x', y='y', color=df['cluster'].astype(str),
        hover_data=['index', 'cluster'],
        title=title
    )
    fig.update_layout(xaxis_title='X', yaxis_title='Y')
    fig.show()


def plot_embedding1(embedding, title="Vector Space Visualization"):
    #fig = px.scatter(x=embedding[:, 0], y=embedding[:, 1], title=title)
    fig = px.scatter(x=range(embedding.shape[0]), y=embedding[:, 1], title=title)
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
    parser.add_argument("--method", default="umap", choices=["umap", "tsne", "pca","none"], help="Dimensionality reduction method")
    parser.add_argument("--dim", type=int, default=1024, help="Dimension")
    parser.add_argument("--search_url", type=str, help="(Optional) HTTP endpoint for vector search")
    parser.add_argument("--query_test", action="store_true", help="Run a test query on first 3 vectors")
    parser.add_argument("--min_size", type=int, default=20, help="Minimum cluster size for HDBSCAN")
    parser.add_argument("--output_file", type=str, default="clusters.json", help="Path to save the output JSON file")
    args = parser.parse_args()

    vectors = load_vectors(args.vector_file, args.dim)
    original_indices = np.arange(len(vectors))  # keep original index mapping
    if args.method == 'none':
        print("No dimensionality reduction selected. Only dbscan")
        labels = find_clusters(vectors, args.min_size) 
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        print(f"Clusters: {n_clusters}")
    else:
        embedding = reduce_dimensions(vectors, args.method)
        print(f"Reduced dimensions: {embedding.shape}")
        labels = find_clusters(embedding, args.min_size) if args.method == 'umap' else [] # [f"{i:06}" for i in original_indices]
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0) if len(labels) > 0 else 0
        print(f"Clusters: {n_clusters}")
        plot_embedding(embedding, labels if len(labels) > 0 else None, original_indices, title=f"{args.method.upper()} Projection")

    df = pd.DataFrame({
        'cluster': labels,
        'index': original_indices
    })
    df.to_json(args.output_file, orient="records", lines=True, index=False)

    if args.search_url and args.query_test:
        print("\nTesting search service with first 3 vectors:")
        for i in range(min(3, len(vectors))):
            print(f"\nQuerying vector {i}")
            result = query_neighbors(vectors[i], args.search_url)
            print(result)

if __name__ == "__main__":
    main()

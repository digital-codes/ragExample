import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns

# Sample Data Generation (replace this with your actual dataset)
data = {
    "Area": ["A", "B", "C", "D", "E"] * 20,
    "Impact": np.random.uniform(50, 100, 100),
    "Cost": np.random.uniform(1, 20, 100),
    "Effort": np.random.uniform(10, 50, 100),
    "Timescale": np.random.uniform(1, 5, 100)
}

# Create DataFrame
df = pd.DataFrame(data)

# Normalize Data for Projection (excluding the 'Area' column)
from sklearn.preprocessing import StandardScaler
features = ["Impact", "Cost", "Effort", "Timescale"]
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df[features])

# PCA for Dimensionality Reduction
pca = PCA(n_components=2)
principal_components = pca.fit_transform(scaled_data)
df_pca = pd.DataFrame(data=principal_components, columns=["PC1", "PC2"])
df_pca["Area"] = df["Area"]

# Variance Explained by Each Principal Component
explained_variance = pca.explained_variance_ratio_
print(f"Explained variance by PC1: {explained_variance[0]:.2f}")
print(f"Explained variance by PC2: {explained_variance[1]:.2f}")


# K-Means Clustering
kmeans = KMeans(n_clusters=5, random_state=42)
df_pca["Cluster"] = kmeans.fit_predict(principal_components)

# Visualization
plt.figure(figsize=(10, 8))
sns.scatterplot(
    x="PC1", y="PC2", hue="Cluster", style="Area", data=df_pca, palette="tab10", s=100
)
plt.title("PCA Projection with K-Means Clustering", fontsize=16)
plt.xlabel(f"Principal Component 1 ({explained_variance[0]*100:.1f}% Variance)")
plt.ylabel(f"Principal Component 2 ({explained_variance[1]*100:.1f}% Variance)")
plt.legend(title="Cluster/Area")
plt.grid(True)
plt.tight_layout()
plt.show()


# Heatmap of Feature Contributions to Principal Components
loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
loading_df = pd.DataFrame(loadings, index=features, columns=["PC1", "PC2"])

plt.figure(figsize=(8, 6))
sns.heatmap(
    loading_df, annot=True, cmap="coolwarm", cbar=True, fmt=".2f"
)
plt.title("Feature Contributions to Principal Components", fontsize=16)
plt.tight_layout()
plt.show()

# Similarity Heatmap
similarity_matrix = cosine_similarity(scaled_data)
plt.figure(figsize=(12, 10))
sns.heatmap(
    similarity_matrix, cmap="viridis", xticklabels=False, yticklabels=False, cbar=True
)
plt.title("Cosine Similarity Heatmap", fontsize=16)
plt.xlabel("Activities")
plt.ylabel("Activities")
plt.tight_layout()
plt.show()



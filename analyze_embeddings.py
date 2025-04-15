import pandas as pd
import numpy as np
import json
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load node community assignment and embeddings
assignments_df = pd.read_csv("output/community_assignments.csv")
embeddings_df = pd.read_csv("output/node_embeddings.csv", index_col="node_id")

# Load community assignments
assignments_df["node_id"] = assignments_df["node_id"].astype(str).str.replace(r"\.0$", "", regex=True)
embeddings_df.index = embeddings_df.index.astype(str).str.replace(r"\.0$", "", regex=True)

embeddings_df = embeddings_df.loc[assignments_df["node_id"]]

# Align embeddings with community assignments
embeddings_df = embeddings_df.loc[assignments_df["node_id"]]
labels = assignments_df["community_id"].values

# Silhouette Score
print("Computing silhouette score...")
sil_score = silhouette_score(embeddings_df.values, labels)
print(f"Silhouette Score: {sil_score:.4f}")

with open("output/silhouette_score.txt", "w") as f:
    f.write(f"Silhouette Score: {sil_score:.4f}\n")

# Export Top Community Nodes
top_n = 10
print(f"Exporting top {top_n} communities' nodes...")
community_sizes = assignments_df["community_id"].value_counts().head(top_n)
top_nodes = []

for comm_id in community_sizes.index:
    nodes = assignments_df[assignments_df["community_id"] == comm_id]["node_id"].tolist()
    top_nodes.append({
        "community_id": comm_id,
        "size": len(nodes),
        "nodes": nodes
    })

with open("output/top_communities_nodes.json", "w") as f:
    json.dump(top_nodes, f, indent=2)
print("Saved to output/top_communities_nodes.json")

# Inter-community Cosine Similarity
print("Computing cosine similarity between community centroids...")
with open("output/community_centroids.json", "r") as f:
    centroids = json.load(f)

centroid_df = pd.DataFrame.from_dict(centroids, orient="index")
similarity_matrix = cosine_similarity(centroid_df.values)

# Save similarity matrix
similarity_df = pd.DataFrame(similarity_matrix, index=centroid_df.index, columns=centroid_df.index)
similarity_df.to_csv("output/inter_community_similarity.csv")
print("Saved to output/inter_community_similarity.csv")

# Heatmap visualization
print("Generating heatmap for community similarity...")
plt.figure(figsize=(10, 8))
sns.heatmap(similarity_df.astype(float), cmap="viridis", square=True, cbar_kws={"label": "Cosine Similarity"})
plt.title("Inter-community Similarity (Cosine)")
plt.tight_layout()
plt.savefig("output/community_similarity_heatmap.png")
plt.close()
print("Saved heatmap to output/community_similarity_heatmap.png")